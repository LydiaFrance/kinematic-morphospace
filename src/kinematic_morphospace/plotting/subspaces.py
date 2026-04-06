"""Plots for individual-vs-shared subspace comparisons (Notebook 08)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Standard hawk colour palette used project-wide
HAWK_COLOURS = {
    "Drogon": "#FC8D62",
    "Rhaegal": "#8DA0CB",
    "Ruby": "#E78AC3",
    "Toothless": "#66C2A5",
    "Charmander": "#A6D854",
}


def plot_method_comparison(method_results, ref_key, max_k, n_comp, colours=None):
    """Plot cosine sweep and reconstruction RMSE comparison for multiple methods.

    Produces a 2-row figure:

    - Top row: minimum principal cosine sweep (k=1..max_k) per method vs
      reference.
    - Bottom row: reconstruction RMSE (k=1..n_comp) per method vs reference.

    Parameters
    ----------
    method_results : dict
        Mapping of method name → dict with keys:
        ``min_cosines`` (array of shape ``(max_k,)``) and
        ``errors`` (array of shape ``(n_comp,)``).
    ref_key : str
        Key in *method_results* for the reference method (plotted in dark grey).
    max_k : int
        Length of the cosine sweep (x-axis for the top row).
    n_comp : int
        Number of components (x-axis for the bottom row).
    colours : dict, optional
        Mapping of method name → hex colour string.  Unrecognised names
        fall back to ``'#51B3D4'``.

    Returns:
    -------
    fig : matplotlib.figure.Figure
    """
    if colours is None:
        colours = {}

    alt_methods = {k: v for k, v in method_results.items() if k != ref_key}
    ref = method_results[ref_key]
    n_alt = len(alt_methods)

    fig, axes = plt.subplots(
        2, n_alt,
        figsize=(14, 6),
        gridspec_kw={"hspace": 0.35, "wspace": 0.3},
    )
    # Ensure axes is always 2-D
    if n_alt == 1:
        axes = axes.reshape(2, 1)

    k_vals_cosine = np.arange(1, max_k + 1)
    k_vals_rmse = np.arange(1, n_comp + 1)

    # Top row: principal cosine sweep
    for j, (name, r) in enumerate(alt_methods.items()):
        ax = axes[0, j]
        ax.plot(k_vals_cosine, ref["min_cosines"], "o-", color="#333333",
                label=ref_key, markersize=4, linewidth=1.5)
        ax.plot(k_vals_cosine, r["min_cosines"], "o-",
                color=colours.get(name, "#51B3D4"),
                label=name, markersize=4, linewidth=1.5)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(0.5, max_k + 0.5)
        ax.set_title(name, fontsize=9)
        ax.axhline(y=0.9, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
        if j == 0:
            ax.set_ylabel("Min principal cosine")
        ax.set_xlabel("k")
        ax.legend(fontsize=6, loc="lower left")

    # Bottom row: reconstruction RMSE
    for j, (name, r) in enumerate(alt_methods.items()):
        ax = axes[1, j]
        ax.plot(k_vals_rmse, ref["errors"], "o-", color="#333333",
                label=ref_key, markersize=4, linewidth=1.5)
        ax.plot(k_vals_rmse, r["errors"], "o-",
                color=colours.get(name, "#51B3D4"),
                label=name, markersize=4, linewidth=1.5)
        ax.set_xlim(0.5, n_comp + 0.5)
        ax.set_title(name, fontsize=9)
        if j == 0:
            ax.set_ylabel("Reconstruction RMSE (m)")
        ax.set_xlabel("k")
        ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    return fig


def plot_cosine_profile(profile, null_mean, null_lo, null_hi, hawk_colours=None):
    """Plot min principal cosine profile with random baseline.

    Parameters
    ----------
    profile : dict
        Mapping of hawk name to 1-D array of min principal cosines
        at each subspace dimension k=1…K.
    null_mean, null_lo, null_hi : array-like
        Mean and 2.5/97.5 percentile bands for the random baseline,
        each of length K.
    hawk_colours : dict, optional
        Mapping of hawk name to colour string.  Defaults to the
        standard project palette.

    Returns:
    -------
    matplotlib.figure.Figure
    """
    if hawk_colours is None:
        hawk_colours = HAWK_COLOURS

    max_k = len(null_mean)
    ks = np.arange(1, max_k + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.fill_between(ks, null_lo, null_hi, color="0.85", label="Random 95% CI")
    ax.plot(ks, null_mean, color="0.55", ls="--", lw=1, label="Random mean")

    for hawk, values in profile.items():
        ax.plot(ks, values, marker="o", ms=4, lw=1.5,
                color=hawk_colours.get(hawk, "0.3"), label=hawk)

    ax.axvline(4, color="0.3", ls=":", lw=1)
    ax.text(4.15, 0.05, "shared core (k = 4)", fontsize=8, color="0.3")

    ax.set_xlabel("Subspace dimension k")
    ax.set_ylabel("Min principal cosine")
    ax.set_xticks(ks)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_title("Principal cosine profile: pooled vs per-hawk subspaces")
    fig.tight_layout()

    return fig


def plot_bootstrap_cosines(replicate_min_cos, max_k=None):
    """Plot bootstrap replicate min principal cosine distributions.

    Parameters
    ----------
    replicate_min_cos : numpy.ndarray
        Array of shape ``(n_replicates, max_k)`` containing the min
        principal cosine for each bootstrap replicate at each subspace
        dimension.
    max_k : int, optional
        Number of subspace dimensions to plot.  Defaults to the number
        of columns in *replicate_min_cos*.

    Returns:
    -------
    matplotlib.figure.Figure
    """
    if max_k is None:
        max_k = replicate_min_cos.shape[1]
    n_reps = replicate_min_cos.shape[0]

    fig, _axes = plt.subplots(1, max_k, figsize=(14, 3), sharey=True, squeeze=False)
    axes = _axes[0]
    for k_idx, ax in enumerate(axes):
        vals = replicate_min_cos[:, k_idx]
        ax.hist(vals, bins=30, color="0.6", edgecolor="0.4")
        ax.axvline(np.percentile(vals, 5),
                   color="C3", ls="--", lw=1.5, label="5th %ile")
        ax.set_title(f"k = {k_idx + 1}", fontsize=9)
        ax.set_xlabel("Min cosine", fontsize=8)
        # Per-panel x limits to show each distribution clearly
        pad = (vals.max() - vals.min()) * 0.15
        ax.set_xlim(vals.min() - pad, vals.max() + pad)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="x", style="plain")
        ax.tick_params(axis="x", labelsize=7)
        if k_idx == 0:
            ax.set_ylabel("Count")
        if k_idx == max_k - 1:
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Bootstrap replicate min principal cosines ({n_reps} replicates)",
        fontsize=10, y=1.04,
    )
    fig.tight_layout()

    return fig
