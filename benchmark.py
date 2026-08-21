"""Fixed forward model, priors and likelihoods for the benchmark."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from emulator import PARAMETER_NAMES as SIGNAL_PARAMETER_NAMES
from emulator import evaluate_21cm


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
TENSOR_DIR = ROOT / "inference_tensors"
DATA_DIR = ROOT / "simulated_data"
N_BETA = 30
N_BEAM = 100
N_SIGNAL = 6
LOG_TWO_PI = jnp.log(2.0 * jnp.pi)

FREQUENCIES_MHZ = jnp.asarray(np.load(TENSOR_DIR / "frequencies_mhz.npy"))
RESPONSE_OPERATOR = jnp.asarray(np.load(TENSOR_DIR / "response_operator.npy"))
with np.load(TENSOR_DIR / "parameter_priors.npz", allow_pickle=False) as _priors:
    PARAMETER_NAMES = tuple(str(name) for name in _priors["names"])
    PRIOR_LOWER = jnp.asarray(_priors["lower"])
    PRIOR_UPPER = jnp.asarray(_priors["upper"])


def load_dataset(index):
    """Load one observation, injected profile and six-dimensional truth."""
    if not 0 <= index < 100:
        raise ValueError("Dataset index must lie between 0 and 99.")
    suffix = f"{index:02d}"
    directory = DATA_DIR / f"dataset_{suffix}"
    return (
        jnp.asarray(np.load(directory / f"observation_{suffix}.npy")),
        jnp.asarray(np.load(directory / f"signal_{suffix}.npy")),
        jnp.asarray(np.load(directory / f"parameters_{suffix}.npy")),
    )


def foreground_terms(beta):
    """Return the mean foreground and prior-whitened beam design matrix."""
    beta = jnp.asarray(beta)
    scaling = jnp.power(FREQUENCIES_MHZ[:, None] / 230.0, -beta[None, :])
    terms = jnp.einsum("ftkr,fr->ftk", RESPONSE_OPERATOR, scaling)
    return terms[:, :, 0], terms[:, :, 1:]


def mean_forward_model(beta, signal_parameters):
    """Return the mean-beam foreground plus global 21-cm signal."""
    foreground, _ = foreground_terms(beta)
    return foreground + evaluate_21cm(signal_parameters)[:, None]


def explicit_forward_model(beta, beam_scores, signal_parameters):
    """Return one explicit beam realisation; beam scores have a unit-normal prior."""
    foreground, design = foreground_terms(beta)
    beam_correction = jnp.einsum("ftm,m->ft", design, beam_scores)
    return foreground + beam_correction + evaluate_21cm(signal_parameters)[:, None]


def marginalised_log_likelihood(parameters, observation):
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


collapsed_log_likelihood = marginalised_log_likelihood


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
