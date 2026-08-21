r"""Fixed radio-galaxy global 21-cm emulator used by the benchmark.

The network learns the scalar mapping

.. math::

   \widehat{T}_{21}(z;\boldsymbol{\theta})
   = f_{\boldsymbol{\phi}}(z,\boldsymbol{\theta}),

where redshift is supplied as an input alongside six continuous astrophysical
parameters. Evaluating the network in parallel over the benchmark's 86
frequencies reconstructs one complete signal profile.

The discrete simulation coordinates are part of the fixed scientific model,
not user options: ``alpha=1.3``, ``nu_0=500 eV`` and ``pop=232``. The public
function therefore accepts exactly six values and always returns the signal in
K on the fixed 50--135 MHz grid.
"""

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
FREQUENCIES_MHZ = np.arange(50.0, 136.0, dtype=np.float32)
_FIXED_ALPHA = 1.3
_FIXED_NU_0 = 500.0
_FIXED_POP = 232.0
_REST_FREQUENCY_MHZ = 1420.405751768


@cache
def _weights():
    """Load the immutable network arrays once and retain them for JAX tracing."""
    with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as data:
        # Saved affine transformations reproduce the coordinates used during
        # training. Network weights remain float32 by construction.
        offsets = np.asarray(data["feature_offset"], dtype=np.float32)
        scales = np.asarray(data["feature_scale"], dtype=np.float32)
        target_scale = np.asarray(data["target_scale"], dtype=np.float32)
        layer_count = int(data["hidden_layer_count"])
        kernels = tuple(
            np.asarray(data[f"hidden_{index}_kernel"], dtype=np.float32)
            for index in range(layer_count)
        )
        biases = tuple(
            np.asarray(data[f"hidden_{index}_bias"], dtype=np.float32)
            for index in range(layer_count)
        )
        readout_kernel = np.asarray(data["readout_kernel"], dtype=np.float32)
        readout_bias = np.asarray(data["readout_bias"], dtype=np.float32)
    return offsets, scales, target_scale, kernels, biases, readout_kernel, readout_bias


def evaluate_21cm(parameters):
    """Evaluate one complete global 21-cm spectrum.

    Parameters
    ----------
    parameters : array-like, shape (6,)
        Values ordered according to ``PARAMETER_NAMES``. Quantities prefixed by
        ``log10`` are supplied in their base-10 logarithmic coordinates.

    Returns
    -------
    jax.Array, shape (86,)
        Brightness-temperature spectrum in K on 50--135 MHz.
    """
    theta = jnp.asarray(parameters, dtype=jnp.float32)
    if theta.shape != (6,):
        raise ValueError(
            f"Expected six parameters in the order {PARAMETER_NAMES}; got {theta.shape}."
        )

    offsets, scales, target_scale, kernels, biases, readout_kernel, readout_bias = (
        jax.tree.map(jnp.asarray, _weights())
    )

    # Restore the complete nine-parameter simulator ordering by inserting the
    # three fixed discrete coordinates around the sampled continuous values.
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
    # Convert observing frequency to redshift, then apply the saved affine
    # transformations to every network feature.
    frequencies_mhz = jnp.asarray(FREQUENCIES_MHZ)
    redshift = _REST_FREQUENCY_MHZ / frequencies_mhz - 1.0
    scaled_redshift = (redshift - offsets[0]) / scales[0]
    scaled_theta = (full_theta - offsets[1:]) / scales[1:]

    # Repeat the same astrophysical realisation at every redshift. The first
    # feature varies along the spectrum; the remaining nine remain fixed.
    features = jnp.concatenate(
        (
            scaled_redshift[:, None],
            jnp.broadcast_to(scaled_theta, (frequencies_mhz.size, scaled_theta.size)),
        ),
        axis=1,
    )

    # Four dense GELU layers map each redshift-parameter row independently to
    # one scalar temperature, allowing the frequency axis to be vectorised.
    values = features
    for kernel, bias in zip(kernels, biases, strict=True):
        values = jax.nn.gelu(values @ kernel + bias)
    # The trained target scale restores mK; divide by 1000 so the forward model
    # and observations use a single temperature unit throughout.
    prediction_mk = (values @ readout_kernel + readout_bias).squeeze(-1)
    return prediction_mk * target_scale / 1000.0


__all__ = ["FREQUENCIES_MHZ", "PARAMETER_NAMES", "evaluate_21cm"]
