r"""Collapsed and full likelihoods for the fixed inference benchmark.

The explicit model is linear in the 100 prior-whitened beam scores,

.. math::

   \mathbf{d} = \boldsymbol{\mu}(\boldsymbol{\phi})
   + \mathbf{H}(\boldsymbol{\beta})\mathbf{z} + \mathbf{n},
   \qquad
   \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
   \quad
   \mathbf{n}\sim\mathcal{N}(\mathbf{0},\sigma_n^2\mathbf{I}).

``Likelihood.evaluate_full`` evaluates this model with :math:`\mathbf{z}`
included in the sampled position. ``Likelihood.evaluate_collapsed`` integrates
:math:`\mathbf{z}` out exactly. Rather than forming the dense 3182-by-3182
effective covariance, the collapsed calculation works in the 100-dimensional
beam space using

.. math::

   \mathbf{Q}=\mathbf{I}+\sigma_n^{-2}\mathbf{H}^{\mathsf T}\mathbf{H},
   \qquad
   \mathbf{h}=\sigma_n^{-2}\mathbf{H}^{\mathsf T}\mathbf{r},

where :math:`\mathbf{r}=\mathbf{d}-\boldsymbol{\mu}`. The resulting likelihood
is mathematically equivalent to using covariance
:math:`\sigma_n^2\mathbf{I}+\mathbf{H}\mathbf{H}^{\mathsf T}`, but requires
only one 100-by-100 Cholesky factorisation.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from .emulator import PARAMETER_NAMES as SIGNAL_PARAMETER_NAMES
from .emulator import evaluate_21cm
from .forward_model import (
    N_BEAM,
    N_BETA,
    N_SIGNAL,
    TENSOR_DIR,
    forward_model,
    foreground_terms,
)


jax.config.update("jax_enable_x64", True)
LOG_TWO_PI = math.log(2.0 * math.pi)

# The directly sampled coordinates are always ordered as 30 foreground
# indices, six signal parameters and one log10 noise amplitude. Their bounds
# are immutable benchmark data rather than runtime configuration.
with np.load(TENSOR_DIR / "parameter_priors.npz", allow_pickle=False) as _priors:
    _PARAMETER_NAMES = tuple(str(name) for name in _priors["names"])
    _PRIOR_LOWER = jnp.asarray(_priors["lower"])
    _PRIOR_UPPER = jnp.asarray(_priors["upper"])


def _collapsed_log_likelihood(parameters, observation):
    r"""Evaluate the 37-dimensional beam-collapsed log likelihood.

    Parameters
    ----------
    parameters : array-like, shape (37,)
        Ordered as ``[beta_00, ..., beta_29, six signal parameters,
        log_noise]``. ``log_noise`` is :math:`\log_{10}\sigma_n` with
        :math:`\sigma_n` measured in K.
    observation : array-like, shape (86, 37)
        Frequency-time observation in K.

    Returns
    -------
    jax.Array
        Scalar log likelihood after exact integration over all 100 beam modes.

    Notes
    -----
    For :math:`N=86\times37` data and whitened beam prior, the implemented
    expression is

    .. math::

       \log\mathcal{L}_{\rm coll}=-\frac12\left[
       \sigma_n^{-2}\mathbf{r}^{\mathsf T}\mathbf{r}
       -\mathbf{h}^{\mathsf T}\mathbf{Q}^{-1}\mathbf{h}
       +N\log\sigma_n^2+\log|\mathbf{Q}|+N\log(2\pi)\right].
    """
    # Separate the foreground, cosmological and noise coordinates. The beam
    # scores are absent because this likelihood integrates them out.
    beta = parameters[:N_BETA]
    signal_parameters = parameters[N_BETA : N_BETA + N_SIGNAL]
    noise_variance = jnp.power(10.0, parameters[-1]) ** 2

    # Construct the mean model and residual while retaining the beam response
    # as a design matrix for the analytical marginalisation below.
    foreground, design = foreground_terms(beta)
    residual = observation - foreground - evaluate_21cm(signal_parameters)[:, None]

    # Flatten frequency and time into N data rows. The beam-mode axis remains
    # last so H has the conventional (N_data, N_beam) matrix layout.
    design = design.reshape((-1, N_BEAM))
    residual = residual.reshape((-1,))

    # Build the posterior beam precision Q and linear term h. Prior whitening
    # makes the prior precision exactly the identity matrix.
    precision = jnp.eye(N_BEAM) + design.T @ design / noise_variance
    rhs = design.T @ residual / noise_variance

    # Cholesky factorisation provides both a stable solve Q^{-1}h and
    # log|Q| = 2 sum(log(diag(L))) without an explicit matrix inverse.
    cholesky = jnp.linalg.cholesky(precision)
    posterior_mean = jax.scipy.linalg.cho_solve((cholesky, True), rhs)

    # Apply the Woodbury correction to the white-noise residual quadratic and
    # the matrix-determinant lemma to its normalisation.
    quadratic = residual @ residual / noise_variance - rhs @ posterior_mean
    logdet = residual.size * jnp.log(noise_variance) + 2.0 * jnp.log(
        jnp.diag(cholesky)
    ).sum()
    return -0.5 * (quadratic + logdet + residual.size * LOG_TWO_PI)


def _full_log_likelihood(parameters, observation):
    r"""Evaluate the 137-dimensional explicit Gaussian log likelihood.

    ``parameters`` is ordered as 30 spectral indices, 100 whitened beam scores,
    six signal parameters and one :math:`\log_{10}\sigma_n`. Conditional on
    these values, every datum is independent with common variance
    :math:`\sigma_n^2`.
    """
    # Extract each contiguous parameter block in the same order used by the
    # prior sampler and saved nested-sampling chains.
    beta = parameters[:N_BETA]
    beam_scores = parameters[N_BETA : N_BETA + N_BEAM]
    signal_start = N_BETA + N_BEAM
    signal_parameters = parameters[signal_start : signal_start + N_SIGNAL]
    noise_variance = jnp.power(10.0, parameters[-1]) ** 2

    # Unlike the collapsed path, the sampled beam realisation is applied
    # directly before evaluating the standard white-noise Gaussian density.
    residual = observation - forward_model(beta, beam_scores, signal_parameters)
    return -0.5 * (
        jnp.sum(residual**2) / noise_variance
        + residual.size * (jnp.log(noise_variance) + LOG_TWO_PI)
    )


def _log_prior(parameters, full=False):
    r"""Evaluate the normalized prior density for either parameterisation.

    The 37 direct parameters have independent uniform priors stored in
    ``parameter_priors.npz``. The explicit model inserts 100 whitened beam
    scores after the foreground block, each distributed as
    :math:`\mathcal{N}(0,1)`.
    """
    if full:
        # Remove the inserted beam block before applying the common uniform
        # bounds, then add its normalized standard-normal log density.
        direct = jnp.concatenate((parameters[:N_BETA], parameters[N_BETA + N_BEAM :]))
        beam_scores = parameters[N_BETA : N_BETA + N_BEAM]
        beam_log_probability = -0.5 * (
            beam_scores @ beam_scores + N_BEAM * LOG_TWO_PI
        )
    else:
        direct = parameters
        beam_log_probability = 0.0

    # Return negative infinity outside the prior support. Inside it, include
    # the uniform normalization so Bayesian evidence retains its proper scale.
    inside = jnp.all((direct >= _PRIOR_LOWER) & (direct <= _PRIOR_UPPER))
    uniform_log_probability = -jnp.log(_PRIOR_UPPER - _PRIOR_LOWER).sum()
    return jnp.where(inside, uniform_log_probability + beam_log_probability, -jnp.inf)


def _sample_prior(key, count, full=False):
    """Draw ``count`` independent live points from the normalized prior.

    Returns shape ``(count, 37)`` for the collapsed problem and
    ``(count, 137)`` for the explicit problem.
    """
    uniform_key, beam_key = jax.random.split(key)

    # Draw every directly sampled parameter in one broadcast operation.
    direct = jax.random.uniform(
        uniform_key,
        (count, _PRIOR_LOWER.size),
        minval=_PRIOR_LOWER,
        maxval=_PRIOR_UPPER,
    )
    if not full:
        return direct

    # Insert standard-normal beam coordinates between the foreground and
    # signal blocks to reproduce the explicit likelihood ordering.
    beam_scores = jax.random.normal(beam_key, (count, N_BEAM), dtype=direct.dtype)
    return jnp.concatenate((direct[:, :N_BETA], beam_scores, direct[:, N_BETA:]), axis=1)


def _names(full=False):
    """Return parameter names in exactly the order consumed by a likelihood."""
    if not full:
        return _PARAMETER_NAMES
    beam_names = tuple(f"beam_{index:03d}" for index in range(N_BEAM))
    return (*_PARAMETER_NAMES[:N_BETA], *beam_names, *SIGNAL_PARAMETER_NAMES, "log_noise")
