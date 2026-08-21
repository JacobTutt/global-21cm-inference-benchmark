r"""Fixed forward model for the global 21-cm inference benchmark.

All expensive sky integration, coordinate rotation and beam decomposition has
already been absorbed into the archived response operator
:math:`\mathcal{R}_{ftkr}`. Its axes are frequency ``f``, observing time ``t``,
response channel ``k`` and foreground region ``r``. Channel zero contains the
mean-beam response; channels 1--100 contain the beam modes after multiplication
by their prior standard deviations.

For regional spectral indices :math:`\boldsymbol{\beta}`, the only foreground
calculation required during inference is

.. math::

   X_{ftk}(\boldsymbol{\beta}) =
   \sum_r \mathcal{R}_{ftkr}
   \left(\frac{\nu_f}{230\,\mathrm{MHz}}\right)^{-\beta_r}.

The mean foreground is :math:`X_{ft0}` and the remaining channels form the
linear design matrix :math:`H_{ftm}=X_{ft,m+1}`. The complete explicit model is

.. math::

   m_{ft} = X_{ft0} + \sum_m H_{ftm}z_m + T_{21,f},
   \qquad \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).

The same terms are consumed by both likelihoods, ensuring that the explicit
and analytically collapsed calculations differ only in their treatment of the
100 beam coefficients.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .emulator import evaluate_21cm


jax.config.update("jax_enable_x64", True)

PACKAGE_ROOT = Path(__file__).resolve().parent
TENSOR_DIR = PACKAGE_ROOT / "inference_tensors"
DATA_DIR = PACKAGE_ROOT / "simulated_data"

# Fixed dimensions of the benchmark. They are deliberately not configurable:
# changing any value would describe a different inference problem.
N_BETA = 30
N_BEAM = 100
N_SIGNAL = 6

# Load the two immutable arrays once. Keeping them as JAX arrays allows every
# contraction below to remain on-device when enclosed by a compiled likelihood.
FREQUENCIES_MHZ = jnp.asarray(np.load(TENSOR_DIR / "frequencies_mhz.npy"))
RESPONSE_OPERATOR = jnp.asarray(np.load(TENSOR_DIR / "response_operator.npy"))


def load_dataset(index):
    """Load one fixed benchmark realisation.

    Parameters
    ----------
    index : int
        Dataset identifier in the inclusive interval 0--99.

    Returns
    -------
    observation : jax.Array, shape (86, 37)
        Frequency-time antenna-temperature data in K.
    signal : jax.Array, shape (86,)
        Injected global 21-cm profile in K. This is retained for validation and
        is never supplied to the likelihood.
    parameters : jax.Array, shape (6,)
        Continuous astrophysical parameters used to generate ``signal``.
    """
    if not 0 <= index < 100:
        raise ValueError("Dataset index must lie between 0 and 99.")

    # Every directory is self-contained so benchmark cases can be copied or
    # inspected independently without a separate manifest.
    suffix = f"{index:02d}"
    directory = DATA_DIR / f"dataset_{suffix}"
    return (
        jnp.asarray(np.load(directory / f"observation_{suffix}.npy")),
        jnp.asarray(np.load(directory / f"signal_{suffix}.npy")),
        jnp.asarray(np.load(directory / f"parameters_{suffix}.npy")),
    )


def foreground_terms(beta):
    r"""Construct the foreground mean and beam design matrix.

    Parameters
    ----------
    beta : array-like, shape (30,)
        Regional foreground spectral indices.

    Returns
    -------
    mean_foreground : jax.Array, shape (86, 37)
        Foreground prediction evaluated with the mean beam.
    design : jax.Array, shape (86, 37, 100)
        Response to each prior-whitened beam coefficient.

    Notes
    -----
    Prior whitening is already contained in ``RESPONSE_OPERATOR``. Therefore
    the explicit beam coordinates have a unit-normal prior and the collapsed
    likelihood uses an identity prior precision.
    """
    beta = jnp.asarray(beta)

    # Evaluate one power law per frequency and foreground region. This is the
    # sole parameter-dependent operation not already folded into the operator.
    scaling = jnp.power(FREQUENCIES_MHZ[:, None] / 230.0, -beta[None, :])

    # Contract the 30 regional responses. Channel zero is the mean response;
    # the remaining channels are columns of the linear beam design matrix.
    terms = jnp.einsum("ftkr,fr->ftk", RESPONSE_OPERATOR, scaling)
    return terms[:, :, 0], terms[:, :, 1:]


def mean_forward_model(beta, signal_parameters):
    r"""Return :math:`\boldsymbol{\mu}=\mathbf{T}_{\rm FG}+\mathbf{T}_{21}`.

    The global signal is independent of observing time, so its 86-frequency
    profile is broadcast across the 37 five-minute spectra.
    """
    foreground, _ = foreground_terms(beta)
    return foreground + evaluate_21cm(signal_parameters)[:, None]


def explicit_forward_model(beta, beam_scores, signal_parameters):
    r"""Evaluate the full model :math:`\boldsymbol{\mu}+\mathbf{H}\mathbf{z}`.

    Parameters
    ----------
    beta : array-like, shape (30,)
        Regional foreground spectral indices.
    beam_scores : array-like, shape (100,)
        Prior-whitened PCA coordinates :math:`\mathbf{z}`. Each coordinate has
        an independent standard-normal prior.
    signal_parameters : array-like, shape (6,)
        Continuous inputs to :func:`global21cm_benchmark.emulator.evaluate_21cm`.

    Returns
    -------
    jax.Array, shape (86, 37)
        Explicit foreground, beam and cosmological prediction in K.
    """
    foreground, design = foreground_terms(beta)

    # Apply one sampled beam realisation to every frequency-time datum before
    # adding the time-independent cosmological signal.
    beam_correction = jnp.einsum("ftm,m->ft", design, beam_scores)
    return foreground + beam_correction + evaluate_21cm(signal_parameters)[:, None]
