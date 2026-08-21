"""Run the fixed Nested Slice Sampling benchmark."""

import argparse
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.utils import finalise, log_weights

from benchmark import (
    full_log_likelihood,
    load_dataset,
    log_prior,
    marginalised_log_likelihood,
    names,
    sample_prior,
)


def run(dataset_index, full=False):
    observation, _, _ = load_dataset(dataset_index)
    likelihood = full_log_likelihood if full else marginalised_log_likelihood
    ndim = 137 if full else 37
    n_live = 25 * ndim
    n_inner = 8 * ndim
    n_delete = int(0.2 * n_live)
    algorithm = blackjax.nss(
        logprior_fn=lambda parameters: log_prior(parameters, full),
        loglikelihood_fn=lambda parameters: likelihood(parameters, observation),
        num_delete=n_delete,
        num_inner_steps=n_inner,
    )

    key = jax.random.PRNGKey(430000)
    key, initial_key = jax.random.split(key)
    state = algorithm.init(sample_prior(initial_key, n_live, full))

    @jax.jit
    def step(state, key):
        key, subkey = jax.random.split(key)
        state, info = algorithm.step(subkey, state)
        return state, key, info

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

    with jax.default_device(cpu):
        final_state = finalise(jax.device_put(state, cpu), dead)
        key, weight_key = jax.random.split(key)
        evidence = jax.scipy.special.logsumexp(
            log_weights(weight_key, final_state, shape=100), axis=0
        )
        slice_calls = sum(
            int(jnp.sum(getattr(info.inner_kernel_info, "info", info.inner_kernel_info).num_steps))
            + int(jnp.sum(getattr(info.inner_kernel_info, "info", info.inner_kernel_info).num_shrink))
            for info in dead
        )

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
