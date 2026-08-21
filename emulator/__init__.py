"""Fixed radio-galaxy 21cmSPACE emulator used by the benchmark."""

from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


PARAMETER_NAMES = (
    "log10fstarII",
    "log10fstarIII",
    "log10Vc",
    "log10fX",
    "tau",
    "log10fradio",
)
FREQUENCIES_MHZ = jnp.arange(50.0, 136.0, dtype=jnp.float32)
_FIXED_ALPHA = 1.3
_FIXED_NU_0 = 500.0
_FIXED_POP = 232.0
_REST_FREQUENCY_MHZ = 1420.405751768


@cache
def _weights():
    with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as data:
        offsets = jnp.asarray(data["feature_offset"], dtype=jnp.float32)
        scales = jnp.asarray(data["feature_scale"], dtype=jnp.float32)
        target_scale = jnp.asarray(data["target_scale"], dtype=jnp.float32)
        layer_count = int(data["hidden_layer_count"])
        kernels = tuple(
            jnp.asarray(data[f"hidden_{index}_kernel"], dtype=jnp.float32)
            for index in range(layer_count)
        )
        biases = tuple(
            jnp.asarray(data[f"hidden_{index}_bias"], dtype=jnp.float32)
            for index in range(layer_count)
        )
        readout_kernel = jnp.asarray(data["readout_kernel"], dtype=jnp.float32)
        readout_bias = jnp.asarray(data["readout_bias"], dtype=jnp.float32)
    return offsets, scales, target_scale, kernels, biases, readout_kernel, readout_bias


def evaluate_21cm(parameters):
    """Return the benchmark 21-cm spectrum in K from six continuous parameters."""
    theta = jnp.asarray(parameters, dtype=jnp.float32)
    if theta.shape != (6,):
        raise ValueError(f"Expected six parameters in the order {PARAMETER_NAMES}; got {theta.shape}.")

    offsets, scales, target_scale, kernels, biases, readout_kernel, readout_bias = _weights()
    full_theta = jnp.stack(
        (
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            jnp.asarray(_FIXED_ALPHA, theta.dtype),
            jnp.asarray(_FIXED_NU_0, theta.dtype),
            theta[4],
            theta[5],
            jnp.asarray(_FIXED_POP, theta.dtype),
        )
    )
    redshift = _REST_FREQUENCY_MHZ / FREQUENCIES_MHZ - 1.0
    scaled_redshift = (redshift - offsets[0]) / scales[0]
    scaled_theta = (full_theta - offsets[1:]) / scales[1:]
    features = jnp.concatenate(
        (
            scaled_redshift[:, None],
            jnp.broadcast_to(scaled_theta, (FREQUENCIES_MHZ.size, scaled_theta.size)),
        ),
        axis=1,
    )

    values = features
    for kernel, bias in zip(kernels, biases, strict=True):
        values = jax.nn.gelu(values @ kernel + bias)
    prediction_mk = (values @ readout_kernel + readout_bias).squeeze(-1)
    return prediction_mk * target_scale / 1000.0


__all__ = ["FREQUENCIES_MHZ", "PARAMETER_NAMES", "evaluate_21cm"]
