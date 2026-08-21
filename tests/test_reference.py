from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from global21cm_benchmark import Dataset, Likelihood, Posterior, Prior, forward_model
from global21cm_benchmark.analysis import create_analysis_plots
from global21cm_benchmark.emulator import PARAMETER_NAMES, evaluate_21cm
from global21cm_benchmark.forward_model import TENSOR_DIR


def test_fixed_benchmark_contract():
    dataset = Dataset(0)
    likelihood = Likelihood(dataset)
    prior = Prior()
    posterior = Posterior(dataset)
    signal = evaluate_21cm(dataset.injected_parameters)
    with np.load(TENSOR_DIR / "reference_evaluation.npz") as reference:
        direct_parameters = jnp.asarray(reference["direct_parameters"])
        full_parameters = jnp.asarray(reference["full_parameters"])
        expected_marginalised = reference["marginalised_log_likelihood"]
        expected_full = reference["full_log_likelihood"]

    evaluate_collapsed = jax.jit(likelihood.evaluate_collapsed)
    evaluate_full = jax.jit(likelihood.evaluate_full)
    marginalised = evaluate_collapsed(direct_parameters)
    full = evaluate_full(full_parameters)
    model = forward_model(
        full_parameters[:30], full_parameters[30:130], full_parameters[130:136]
    )
    collapsed_log_prior = prior.evaluate_collapsed(direct_parameters)
    full_log_prior = prior.evaluate_full(full_parameters)

    assert dataset.observation.shape == (86, 37)
    assert model.shape == dataset.observation.shape
    assert jnp.max(jnp.abs(signal - dataset.injected_signal)) < 1e-6
    assert jnp.allclose(marginalised, expected_marginalised, rtol=1e-8)
    assert jnp.allclose(full, expected_full, rtol=1e-8)
    assert jnp.allclose(
        posterior.evaluate_collapsed(direct_parameters),
        marginalised + collapsed_log_prior,
    )
    assert jnp.allclose(
        posterior.evaluate_full(full_parameters),
        full + full_log_prior,
    )


def test_analysis_outputs():
    dataset = Dataset(0)
    particles = np.asarray(Prior().sample_collapsed(jax.random.key(7), 512))
    parameter_names = np.asarray(
        [*(f"beta_{index:02d}" for index in range(30)), *PARAMETER_NAMES, "log_noise"]
    )
    weights = np.full(len(particles), 1.0 / len(particles))

    with TemporaryDirectory() as directory:
        results_file = f"{directory}/nested_sampling_results.npz"
        np.savez(
            results_file,
            particles=particles,
            parameter_names=parameter_names,
            posterior_weights=weights,
        )
        corner_file, recovery_file = create_analysis_plots(results_file, dataset)
        assert corner_file.stat().st_size > 0
        assert recovery_file.stat().st_size > 0


if __name__ == "__main__":
    test_fixed_benchmark_contract()
    test_analysis_outputs()
    print("reference check passed")
