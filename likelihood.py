"""Collapsed and full likelihoods for the fixed inference benchmark."""

import jax
import jax.numpy as jnp
import numpy as np

from emulator import PARAMETER_NAMES as SIGNAL_PARAMETER_NAMES
from emulator import evaluate_21cm
from forward_model import (
    N_BEAM,
    N_BETA,
    N_SIGNAL,
    TENSOR_DIR,
    explicit_forward_model,
    foreground_terms,
)


jax.config.update("jax_enable_x64", True)
LOG_TWO_PI = jnp.log(2.0 * jnp.pi)

with np.load(TENSOR_DIR / "parameter_priors.npz", allow_pickle=False) as _priors:
    PARAMETER_NAMES = tuple(str(name) for name in _priors["names"])
    PRIOR_LOWER = jnp.asarray(_priors["lower"])
    PRIOR_UPPER = jnp.asarray(_priors["upper"])


def collapsed_log_likelihood(parameters, observation):
    """Integrate the 100 whitened beam coefficients out analytically."""
    beta = parameters[:N_BETA]
    signal_parameters = parameters[N_BETA : N_BETA + N_SIGNAL]
    noise_variance = jnp.power(10.0, parameters[-1]) ** 2
    foreground, design = foreground_terms(beta)
    residual = observation - foreground - evaluate_21cm(signal_parameters)[:, None]
    design = design.reshape((-1, N_BEAM))
    residual = residual.reshape((-1,))

    precision = jnp.eye(N_BEAM) + design.T @ design / noise_variance
    rhs = design.T @ residual / noise_variance
    cholesky = jnp.linalg.cholesky(precision)
    posterior_mean = jax.scipy.linalg.cho_solve((cholesky, True), rhs)
    quadratic = residual @ residual / noise_variance - rhs @ posterior_mean
    logdet = residual.size * jnp.log(noise_variance) + 2.0 * jnp.log(
        jnp.diag(cholesky)
    ).sum()
    return -0.5 * (quadratic + logdet + residual.size * LOG_TWO_PI)


def full_log_likelihood(parameters, observation):
    """Evaluate the Gaussian likelihood with all 100 beam scores explicit."""
    beta = parameters[:N_BETA]
    beam_scores = parameters[N_BETA : N_BETA + N_BEAM]
    signal_start = N_BETA + N_BEAM
    signal_parameters = parameters[signal_start : signal_start + N_SIGNAL]
    noise_variance = jnp.power(10.0, parameters[-1]) ** 2
    residual = observation - explicit_forward_model(beta, beam_scores, signal_parameters)
    return -0.5 * (
        jnp.sum(residual**2) / noise_variance
        + residual.size * (jnp.log(noise_variance) + LOG_TWO_PI)
    )


def log_prior(parameters, full=False):
    """Evaluate the fixed uniform priors and, when explicit, unit-normal beam prior."""
    if full:
        direct = jnp.concatenate((parameters[:N_BETA], parameters[N_BETA + N_BEAM :]))
        beam_scores = parameters[N_BETA : N_BETA + N_BEAM]
        beam_log_probability = -0.5 * (
            beam_scores @ beam_scores + N_BEAM * LOG_TWO_PI
        )
    else:
        direct = parameters
        beam_log_probability = 0.0
    inside = jnp.all((direct >= PRIOR_LOWER) & (direct <= PRIOR_UPPER))
    uniform_log_probability = -jnp.log(PRIOR_UPPER - PRIOR_LOWER).sum()
    return jnp.where(inside, uniform_log_probability + beam_log_probability, -jnp.inf)


def sample_prior(key, count, full=False):
    """Draw initial live points from the benchmark prior."""
    uniform_key, beam_key = jax.random.split(key)
    direct = jax.random.uniform(
        uniform_key, (count, PRIOR_LOWER.size), minval=PRIOR_LOWER, maxval=PRIOR_UPPER
    )
    if not full:
        return direct
    beam_scores = jax.random.normal(beam_key, (count, N_BEAM), dtype=direct.dtype)
    return jnp.concatenate((direct[:, :N_BETA], beam_scores, direct[:, N_BETA:]), axis=1)


def names(full=False):
    """Return the fixed parameter ordering used by saved chains."""
    if not full:
        return PARAMETER_NAMES
    beam_names = tuple(f"beam_{index:03d}" for index in range(N_BEAM))
    return (*PARAMETER_NAMES[:N_BETA], *beam_names, *SIGNAL_PARAMETER_NAMES, "log_noise")
