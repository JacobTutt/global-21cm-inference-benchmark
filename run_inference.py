r"""Run the fixed Nested Slice Sampling benchmark.

Nested sampling maintains a set of live points drawn from the prior subject to
an increasing likelihood constraint. At each iteration the lowest-likelihood
points are recorded as dead points and replaced through constrained slice
sampling. The run terminates when

.. math::

   \Delta\log Z = \log Z_{\rm live} - \log Z < -3,

so the remaining live-point contribution is negligible at the benchmark's
chosen evidence tolerance.

Only the dataset index and whether the beam coefficients are sampled explicitly
are selectable. Live-point count, slice-step budget, deletion fraction, random
seed and stopping condition are fixed to the values used for the benchmark.
"""

import argparse
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.utils import finalise, log_weights

from forward_model import load_dataset
from likelihood import (
    collapsed_log_likelihood,
    full_log_likelihood,
    log_prior,
    names,
    sample_prior,
)


def run(dataset_index, full=False):
    """Run one fixed NSS fit and save its samples, evidence and call count.

    Parameters
    ----------
    dataset_index : int
        Benchmark realisation in the inclusive interval 0--99.
    full : bool, default=False
        If false, run the 37-dimensional beam-collapsed likelihood. If true,
        run the 137-dimensional likelihood with explicit beam coefficients.
    """
    # The injected signal and generating parameters are intentionally ignored:
    # inference receives only the simulated observation.
    observation, _, _ = load_dataset(dataset_index)
    likelihood = full_log_likelihood if full else collapsed_log_likelihood

    # Paper configuration: 25 live points and eight constrained slice steps per
    # dimension, replacing 20% of the live set in each outer NS iteration.
    ndim = 137 if full else 37
    n_live = 25 * ndim
    n_inner = 8 * ndim
    n_delete = int(0.2 * n_live)

    # Both likelihoods share the same NSS implementation and differ only in
    # their parameter dimension, prior and treatment of beam uncertainty.
    algorithm = blackjax.nss(
        logprior_fn=lambda parameters: log_prior(parameters, full),
        loglikelihood_fn=lambda parameters: likelihood(parameters, observation),
        num_delete=n_delete,
        num_inner_steps=n_inner,
    )

    # Draw and evaluate the complete initial live set in parallel.
    key = jax.random.PRNGKey(430000)
    key, initial_key = jax.random.split(key)
    state = algorithm.init(sample_prior(initial_key, n_live, full))

    @jax.jit
    def step(state, key):
        """Perform one compiled delete-and-replace NSS transition."""
        key, subkey = jax.random.split(key)
        state, info = algorithm.step(subkey, state)
        return state, key, info

    # Dead-point batches are accumulated on the accelerator for efficient
    # transitions, then periodically migrated to CPU memory to bound GPU use.
    dead = []
    pending = []
    iteration = 0
    cpu = jax.devices("cpu")[0]
    while float(state.logZ_live - state.logZ) >= -3.0:
        state, key, info = step(state, key)
        pending.append(info)
        iteration += 1
        if iteration % 10 == 0:
            print(
                f"dead={iteration * n_delete:,} "
                f"dlogZ={float(state.logZ_live - state.logZ):.3f}",
                flush=True,
            )
        if len(pending) * n_delete >= 100_000:
            dead.extend(jax.device_put(pending, cpu))
            pending.clear()
    if pending:
        dead.extend(jax.device_put(pending, cpu))

    # Finalisation appends the surviving live points to the recorded dead-point
    # sequence. It is intentionally performed on CPU to avoid a final GPU
    # memory spike from the complete chain.
    with jax.default_device(cpu):
        final_state = finalise(jax.device_put(state, cpu), dead)

        # Bootstrap the stochastic nested-sampling weights 100 times to retain
        # the evidence uncertainty induced by the unknown prior-volume shrinkage.
        key, weight_key = jax.random.split(key)
        evidence = jax.scipy.special.logsumexp(
            log_weights(weight_key, final_state, shape=100), axis=0
        )

        # Each stepping-out and shrinkage proposal evaluates the likelihood.
        # Recording this total makes sampler efficiency directly comparable.
        slice_calls = sum(
            int(jnp.sum(getattr(info.inner_kernel_info, "info", info.inner_kernel_info).num_steps))
            + int(jnp.sum(getattr(info.inner_kernel_info, "info", info.inner_kernel_info).num_shrink))
            for info in dead
        )

    # Store plain NumPy arrays so analysis does not require the sampler package.
    label = "full" if full else "marginalised"
    output = Path("results") / f"dataset_{dataset_index:02d}" / label
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "nested_sampling_results.npz",
        parameter_names=np.asarray(names(full)),
        particles=np.asarray(final_state.particles),
        log_likelihood=np.asarray(final_state.loglikelihood),
        log_likelihood_birth=np.asarray(final_state.loglikelihood_birth),
        log_evidence=np.asarray(evidence),
        total_likelihood_calls=np.asarray(n_live + slice_calls),
    )
    print(f"saved {output / 'nested_sampling_results.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=int, help="Dataset index from 0 to 99.")
    parser.add_argument("--full", action="store_true", help="Sample the 100 beam scores explicitly.")
    arguments = parser.parse_args()
    run(arguments.dataset, arguments.full)
