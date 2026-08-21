"""Fixed forward model for the global 21-cm inference benchmark."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from emulator import evaluate_21cm


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
TENSOR_DIR = ROOT / "inference_tensors"
DATA_DIR = ROOT / "simulated_data"
N_BETA = 30
N_BEAM = 100
N_SIGNAL = 6

FREQUENCIES_MHZ = jnp.asarray(np.load(TENSOR_DIR / "frequencies_mhz.npy"))
RESPONSE_OPERATOR = jnp.asarray(np.load(TENSOR_DIR / "response_operator.npy"))


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
