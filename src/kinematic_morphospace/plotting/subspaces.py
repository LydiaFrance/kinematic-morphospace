"""Plots for individual-vs-shared subspace comparisons (Notebook 08)."""

import matplotlib.pyplot as plt
import numpy as np
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
    """Compare cosine sweep and reconstruction RMSE across alternative PCA methods.

    Produces a 2-row figure. The top row shows the minimum principal cosine
    between each method's subspace and the reference at increasing subspace
    dimension k=1..max_k. The bottom row shows the reconstruction RMSE at
    k=1..n_comp. This reveals which alternative methods best preserve the
    shared morphospace structure.

    Args:
        method_results: Dict mapping method name to a dict with keys
            'min_cosines' (array of length max_k) and 'errors' (array of
            length n_comp).
        ref_key: Key in method_results identifying the reference method, drawn
            in dark grey for comparison.
        max_k: Number of subspace dimensions for the cosine sweep (x-axis of
            the top row).
        n_comp: Number of components for the RMSE plot (x-axis of the bottom row).
        colours: Optional dict mapping method name to hex colour string.
            Unrecognised names fall back to '#51B3D4'. Defaults to None.

    Returns:
        Figure containing the 2-row method comparison.
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
    """Plot minimum principal cosine profiles comparing pooled and per-hawk subspaces.

    Shows how well each individual hawk's k-dimensional subspace aligns with the
    pooled PCA subspace as k increases. Values above the random baseline indicate
    that the individual and pooled solutions share genuine structure. A vertical
    line marks the shared core dimension (k=4).

    Args:
        profile: Dict mapping hawk name to a 1-D array of minimum principal
            cosines at each subspace dimension k=1…K.
        null_mean: Mean of the random-baseline distribution, length K.
        null_lo: 2.5th percentile of the random baseline, length K.
        null_hi: 97.5th percentile of the random baseline, length K.
        hawk_colours: Optional dict mapping hawk name to colour string. Defaults
            to the standard project palette (HAWK_COLOURS).

    Returns:
        Figure containing the cosine profile plot.
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
    """Plot bootstrap replicate distributions of the minimum principal cosine.

    Creates one histogram panel per subspace dimension k=1…max_k, showing the
    spread of minimum cosine values across bootstrap replicates. The 5th
    percentile is marked to indicate the lower confidence bound. Useful for
    assessing whether the shared subspace structure is stable across resamples.

    Args:
        replicate_min_cos: Array of shape (n_replicates, max_k) containing the
            minimum principal cosine for each replicate at each k.
        max_k: Number of subspace dimensions to plot. Defaults to the number of
            columns in replicate_min_cos.

    Returns:
        Figure containing the bootstrap cosine distribution plots.
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
