"""Posterior diagnostics generated after each Nested Slice Sampling run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .api import Dataset
from .emulator import FREQUENCIES_MHZ, PARAMETER_NAMES, evaluate_21cm
from .likelihood import _PRIOR_LOWER, _PRIOR_UPPER


_SIGNAL_LABELS = (
    r"$\log_{10} f_{\star,\mathrm{II}}$",
    r"$\log_{10} f_{\star,\mathrm{III}}$",
    r"$\log_{10} V_{\mathrm{c}}$",
    r"$\log_{10} f_{\mathrm{X}}$",
    r"$\tau$",
    r"$\log_{10} f_{\mathrm{radio}}$",
)
_POSTERIOR_COLOR = "#b2182b"
_PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ("STIXGeneral", "DejaVu Serif"),
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
}


def _weighted_quantiles(values, weights, probabilities):
    """Return weighted quantiles independently for every spectrum channel."""
    quantiles = np.empty((len(probabilities), values.shape[1]))
    for channel in range(values.shape[1]):
        order = np.argsort(values[:, channel])
        cumulative = np.cumsum(weights[order])
        quantiles[:, channel] = np.interp(
            probabilities, cumulative, values[order, channel]
        )
    return quantiles


def create_analysis_plots(results_file, dataset):
    """Create the signal-parameter corner and signal-recovery plots.

    Parameters
    ----------
    results_file : path-like
        ``nested_sampling_results.npz`` produced by the reference workflow.
    dataset : Dataset
        Dataset fitted by that run. Its injected signal and parameters are used
        only as truth markers after inference has completed.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to ``signal_corner.png`` and ``signal_recovery.png``.

    Notes
    -----
    The posterior weights average over the same stochastic prior-volume
    shrinkage realisations used for the evidence calculation. Credible bands
    and corner contours therefore retain the Nested Sampling weights rather
    than treating dead points as equally probable posterior samples.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("create_analysis_plots expects a Dataset instance.")

    results_file = Path(results_file)
    with np.load(results_file, allow_pickle=False) as results:
        particles = np.asarray(results["particles"])
        parameter_names = tuple(str(name) for name in results["parameter_names"])
        weights = np.asarray(results["posterior_weights"], dtype=float)

    if particles.ndim != 2 or weights.shape != (particles.shape[0],):
        raise ValueError("Particles and posterior weights have incompatible shapes.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Posterior weights must be finite, non-negative, and non-zero.")
    weights /= weights.sum()

    # Locate the signal block by name so this works for both the 37-dimensional
    # collapsed chain and the 137-dimensional explicit-beam chain.
    try:
        signal_indices = [parameter_names.index(name) for name in PARAMETER_NAMES]
    except ValueError as error:
        raise ValueError(
            "Nested Sampling results do not contain all signal parameters."
        ) from error
    signal_parameters = particles[:, signal_indices]
    truth = np.asarray(dataset.injected_parameters)
    signal_ranges = list(
        zip(
            np.asarray(_PRIOR_LOWER[30:36], dtype=float),
            np.asarray(_PRIOR_UPPER[30:36], dtype=float),
            strict=True,
        )
    )

    output_directory = results_file.parent
    corner_file = output_directory / "signal_corner.png"
    recovery_file = output_directory / "signal_recovery.png"

    with plt.rc_context(_PLOT_STYLE):
        figure = corner.corner(
            signal_parameters,
            weights=weights,
            labels=_SIGNAL_LABELS,
            truths=truth,
            truth_color="black",
            color=_POSTERIOR_COLOR,
            range=signal_ranges,
            bins=35,
            smooth=1.0,
            smooth1d=1.0,
            levels=(0.68, 0.95),
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            contour_kwargs={"linewidths": 1.0},
            hist_kwargs={"linewidth": 1.4},
            label_kwargs={"fontsize": 11},
        )
        figure.set_size_inches(12, 12)
        figure.suptitle(
            f"Dataset {dataset.index:02d}", fontsize=17, fontweight="bold", y=0.995
        )
        figure.savefig(corner_file, dpi=200, bbox_inches="tight")
        plt.close(figure)

        # Evaluate the emulator in bounded batches so analysis remains usable
        # on CPU as well as on the benchmark GPU.
        evaluate_batch = jax.jit(jax.vmap(evaluate_21cm))
        signal_profiles = []
        for start in range(0, len(signal_parameters), 4096):
            batch = evaluate_batch(jnp.asarray(signal_parameters[start : start + 4096]))
            signal_profiles.append(np.asarray(batch) * 1000.0)
        signal_profiles = np.concatenate(signal_profiles)

        posterior_mean = np.average(signal_profiles, axis=0, weights=weights)
        lower_95, lower_68, upper_68, upper_95 = _weighted_quantiles(
            signal_profiles, weights, (0.025, 0.16, 0.84, 0.975)
        )
        injected = np.asarray(dataset.injected_signal) * 1000.0
        rmse = np.sqrt(np.mean((posterior_mean - injected) ** 2))

        figure, axis = plt.subplots(figsize=(7.2, 5.0))
        band_95 = axis.fill_between(
            FREQUENCIES_MHZ,
            lower_95,
            upper_95,
            color="#f6dadd",
            label="95% credible interval",
        )
        band_68 = axis.fill_between(
            FREQUENCIES_MHZ,
            lower_68,
            upper_68,
            color="#e9a5ae",
            label="68% credible interval",
        )
        (injected_line,) = axis.plot(
            FREQUENCIES_MHZ, injected, color="black", linewidth=1.8, label="Injected"
        )
        (mean_line,) = axis.plot(
            FREQUENCIES_MHZ,
            posterior_mean,
            color=_POSTERIOR_COLOR,
            linewidth=2.0,
            label="Posterior mean",
        )
        axis.set(
            xlabel="Frequency [MHz]",
            ylabel=r"$T_{21}$ [mK]",
            title=f"Dataset {dataset.index:02d} | RMSE {rmse:.1f} mK",
        )
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(
            [injected_line, mean_line, band_68, band_95],
            [
                "Injected",
                "Posterior mean",
                "68% credible interval",
                "95% credible interval",
            ],
            loc="best",
            frameon=False,
        )
        figure.tight_layout()
        figure.savefig(recovery_file, dpi=200, bbox_inches="tight")
        plt.close(figure)

    return corner_file, recovery_file


__all__ = ["create_analysis_plots"]
