import jax
import jax.numpy as jnp
import numpy as np

from global21cm_benchmark.emulator import evaluate_21cm
from global21cm_benchmark.forward_model import TENSOR_DIR, load_dataset
from global21cm_benchmark.likelihood import (
    collapsed_log_likelihood,
    full_log_likelihood,
)


def test_fixed_benchmark_contract():
    observation, injected_signal, truth = load_dataset(0)
    signal = evaluate_21cm(truth)
    with np.load(TENSOR_DIR / "reference_evaluation.npz") as reference:
        direct_parameters = jnp.asarray(reference["direct_parameters"])
        full_parameters = jnp.asarray(reference["full_parameters"])
        expected_marginalised = reference["marginalised_log_likelihood"]
        expected_full = reference["full_log_likelihood"]

    marginalised = jax.jit(collapsed_log_likelihood)(direct_parameters, observation)
    full = jax.jit(full_log_likelihood)(full_parameters, observation)

    assert observation.shape == (86, 37)
    assert jnp.max(jnp.abs(signal - injected_signal)) < 1e-6
    assert jnp.allclose(marginalised, expected_marginalised, rtol=1e-8)
    assert jnp.allclose(full, expected_full, rtol=1e-8)


if __name__ == "__main__":
    test_fixed_benchmark_contract()
    print("reference check passed")
