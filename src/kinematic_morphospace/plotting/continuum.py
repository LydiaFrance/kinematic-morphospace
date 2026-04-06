"""Plotting for flight-behaviour continuum analysis (NB12).

These helpers take arrays/DataFrames computed in the notebook and return
matplotlib figures, so the calculations stay visible in the notebook
while the drawing code is factored out here.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde


def compute_flight_phase_traces(
    scores: np.ndarray,
    phase: np.ndarray,
    flight_phase_map: Mapping[str, int],
    *,
    bin_value: np.ndarray,
    include_mask: np.ndarray,
    bin_edges: np.ndarray,
    min_count: int = 50,
) -> dict[str, "pd.DataFrame | None"]:
    """Bin PC1/PC2 scores by a continuous variable within each flight phase.

    Divides bin_value into the supplied bins, then for each flight phase
    computes the mean and standard deviation of PC1 and PC2 within each bin.
    Bins with fewer than min_count samples are dropped to avoid noisy estimates.

    Args:
        scores: PCA score array of shape (n_frames, n_components).
        phase: Integer phase code per frame, aligned with scores.
        flight_phase_map: Dict mapping phase name (str) to its integer code.
        bin_value: Continuous variable to bin by (e.g. distance to perch),
            shape (n_frames,).
        include_mask: Boolean mask of frames to include (e.g. excluding perch
            grab frames), shape (n_frames,).
        bin_edges: Edges of the bins for bin_value, length n_bins + 1.
        min_count: Minimum number of frames per bin to retain. Defaults to 50.

    Returns:
        Dict mapping each phase name to a DataFrame indexed by bin midpoint with
        columns n, PC1_mean, PC1_std, PC2_mean, PC2_std, or None if no frames
        exist for that phase.
    """
    bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(bin_value, bin_edges) - 1
    valid_bin = (bin_idx >= 0) & (bin_idx < len(bin_mid))

    traces: dict[str, "pd.DataFrame | None"] = {}
    for name, code in flight_phase_map.items():
        m = (phase == code) & include_mask & valid_bin
        if not m.any():
            traces[name] = None
            continue
        df = pd.DataFrame({
            "bin": bin_idx[m],
            "PC1": scores[m, 0],
            "PC2": scores[m, 1],
        }).groupby("bin").agg(
            n=("PC1", "count"),
            PC1_mean=("PC1", "mean"), PC1_std=("PC1", "std"),
            PC2_mean=("PC2", "mean"), PC2_std=("PC2", "std"),
        )
        df = df[df["n"] >= min_count]
        df = df.set_axis(bin_mid[df.index.to_numpy()], axis=0)
        traces[name] = df
    return traces


def _draw_kde_contours(ax: Axes, xy: np.ndarray, colour: str,
                       *, n_levels: int = 5, linewidth: float = 1.0) -> None:
    """Draw KDE contours from a (2, n) array of paired (x, y) samples."""
    kde = gaussian_kde(xy)
    pad = 0.05
    xmin, xmax = np.percentile(xy[0], [0.5, 99.5])
    ymin, ymax = np.percentile(xy[1], [0.5, 99.5])
    xg, yg = np.mgrid[xmin - pad:xmax + pad:120j, ymin - pad:ymax + pad:120j]
    z = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    ax.contour(xg, yg, z, levels=n_levels, colors=colour,
               linewidths=linewidth, alpha=0.9)


def _scatter_flight_phase(ax: Axes, scores: np.ndarray, idx: np.ndarray, colour: str,
                   *, alpha: float = 0.15, s: float = 1.0,
                   label: str | None = None) -> None:
    ax.scatter(scores[idx, 0], scores[idx, 1],
               s=s, alpha=alpha, c=colour, label=label, rasterized=True)


def plot_flight_phase_overlay(
    scores: np.ndarray,
    flap_idx: np.ndarray,
    glide_idx: np.ndarray,
    flap_kde_idx: np.ndarray,
    glide_kde_idx: np.ndarray,
    *,
    colours: Mapping[str, str],
    flap_label: str = "Flapping",
    glide_label: str = "Gliding",
    title: str = "Flapping vs gliding in PC1-PC2 kinematic morphospace",
    xlabel: str = "PC1 score",
    ylabel: str = "PC2 score",
    alpha: float = 0.15,
) -> Figure:
    """PC1-PC2 scatter plot of flapping versus gliding frames with KDE contours.

    Overlays translucent scatter points for each flight phase and draws KDE
    contour lines to show the density of each phase in the morphospace.
    Gliding occupies a concentrated subset of the flapping region, consistent
    with a continuum rather than discrete gaits.

    Args:
        scores: PCA score array of shape (n_frames, n_components).
        flap_idx: Row indices of flapping frames in scores.
        glide_idx: Row indices of gliding frames in scores.
        flap_kde_idx: Subset of flap_idx used for KDE estimation (may be
            downsampled for speed).
        glide_kde_idx: Subset of glide_idx used for KDE estimation.
        colours: Dict with keys 'flapping' and 'gliding' mapping to hex colour
            strings.
        flap_label: Legend label for flapping frames. Defaults to 'Flapping'.
        glide_label: Legend label for gliding frames. Defaults to 'Gliding'.
        title: Plot title. Defaults to a standard description.
        xlabel: X-axis label. Defaults to 'PC1 score'.
        ylabel: Y-axis label. Defaults to 'PC2 score'.
        alpha: Scatter-point opacity. Defaults to 0.15.

    Returns:
        Figure containing the overlay plot.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _scatter_flight_phase(ax, scores, flap_idx, colours["flapping"],
                   alpha=alpha, label=flap_label)
    _scatter_flight_phase(ax, scores, glide_idx, colours["gliding"],
                   alpha=alpha, label=glide_label)
    _draw_kde_contours(ax, scores[flap_kde_idx, :2].T, colours["flapping"])
    _draw_kde_contours(ax, scores[glide_kde_idx, :2].T, colours["gliding"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, markerscale=5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_transition_overlay(
    scores: np.ndarray,
    flap_idx: np.ndarray,
    glide_idx: np.ndarray,
    trans_idx: np.ndarray,
    *,
    n_transition: int,
    colours: Mapping[str, str],
    title: str = "Transition frames overlaid on the two-phase comparison",
    xlabel: str = "PC1 score  (wings down / up)",
    ylabel: str = "PC2 score  (folded / spread)",
) -> Figure:
    """PC1-PC2 scatter showing transition frames over a fainter flapping/gliding backdrop.

    Plots flapping and gliding frames at low opacity as context, then overlays
    transition frames at higher opacity. This reveals where in the morphospace
    the transition between gaits occurs — typically not a sharp boundary.

    Args:
        scores: PCA score array of shape (n_frames, n_components).
        flap_idx: Row indices of flapping frames in scores.
        glide_idx: Row indices of gliding frames in scores.
        trans_idx: Row indices of transition frames in scores.
        n_transition: Total count of transition frames, used for the legend label.
        colours: Dict with keys 'flapping', 'gliding', and 'transition' mapping
            to hex colour strings.
        title: Plot title. Defaults to a standard description.
        xlabel: X-axis label. Defaults to a wing-position description.
        ylabel: Y-axis label. Defaults to a wing-spread description.

    Returns:
        Figure containing the transition overlay plot.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _scatter_flight_phase(ax, scores, flap_idx, colours["flapping"],
                   alpha=0.08, label="Flapping")
    _scatter_flight_phase(ax, scores, glide_idx, colours["gliding"],
                   alpha=0.08, label="Gliding")
    ax.scatter(scores[trans_idx, 0], scores[trans_idx, 1],
               s=1.5, alpha=0.35, c=colours["transition"],
               label=f"Transition (n={n_transition:,})", rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, markerscale=5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_flight_phase_time_traces(
    traces: Mapping[str, Any],
    flight_phase_colours: Mapping[str, str],
) -> Figure:
    """Three-panel figure of binned PC1/PC2 mean traces and PC1 spread versus distance to perch.

    Panel (a) shows PC1 mean ± 1 SD per flight phase; panel (b) shows PC2 mean
    ± 1 SD; panel (c) shows within-bin PC1 standard deviation as a proxy for
    wingbeat activity. Together these reveal how wing-shape kinematics evolve
    during the approach.

    Args:
        traces: Dict mapping flight phase name to a DataFrame (or None if the
            phase has no data). Each DataFrame is indexed by bin midpoint and has
            columns PC1_mean, PC1_std, PC2_mean, PC2_std, n.
        flight_phase_colours: Dict mapping phase name to a hex colour string.

    Returns:
        Figure containing the three stacked panels.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # (a) PC1 mean +/- std
    ax = axes[0]
    for name, g in traces.items():
        if g is None:
            continue
        c = flight_phase_colours[name]
        ax.fill_between(g.index, g["PC1_mean"] - g["PC1_std"],
                        g["PC1_mean"] + g["PC1_std"],
                        color=c, alpha=0.2, linewidth=0)
        ax.plot(g.index, g["PC1_mean"], color=c, linewidth=1.5, label=name)
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
    ax.set_ylabel("PC1 (wings down  <->  up)")
    ax.set_title("(a) PC1 vs distance to perch (mean +/- 1 SD)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) PC2 mean +/- std
    ax = axes[1]
    for name, g in traces.items():
        if g is None:
            continue
        c = flight_phase_colours[name]
        ax.fill_between(g.index, g["PC2_mean"] - g["PC2_std"],
                        g["PC2_mean"] + g["PC2_std"],
                        color=c, alpha=0.2, linewidth=0)
        ax.plot(g.index, g["PC2_mean"], color=c, linewidth=1.5, label=name)
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.3)
    ax.set_ylabel("PC2 (folded  <->  spread)")
    ax.set_title("(b) PC2 vs distance to perch  (mean +/- 1 SD)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) Within-bin std of PC1 = wingbeat-activity indicator
    ax = axes[2]
    for name, g in traces.items():
        if g is None:
            continue
        ax.plot(g.index, g["PC1_std"], color=flight_phase_colours[name],
                linewidth=1.8, label=name)
    ax.set_xlabel("Distance to perch (m)")
    ax.set_ylabel("Within-bin std of PC1")
    ax.set_title("(c) Within-bin PC1 spread = wingbeat indicator")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()

    fig.tight_layout()
    return fig


def plot_bic_sweep(
    k_values: Sequence[int],
    bic_curve: np.ndarray,
    *,
    best_k: int,
    n_components: int,
    n_samples: int,
    colours: Mapping[str, str],
) -> Figure:
    """BIC curve for GMM model selection, annotated with k=2 and BIC-minimum markers.

    Shows how the Bayesian Information Criterion varies with the number of GMM
    components. If the data supported discrete gaits, a clear elbow at k=2 or
    k=3 would be expected; the absence of such an elbow supports a continuum
    interpretation.

    Args:
        k_values: Sequence of k values at which the GMM was fitted.
        bic_curve: BIC score for each k, aligned with k_values.
        best_k: Value of k at which BIC is minimised, marked with a vertical line.
        n_components: Number of PCA components used for GMM fitting (for title).
        n_samples: Number of frames in the subsample used for fitting (for title).
        colours: Dict with at least 'flapping' and 'gliding' keys mapping to hex
            colour strings, used for the vertical-line annotations.

    Returns:
        Figure containing the BIC sweep plot.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(list(k_values), bic_curve, "ko-", markersize=3)
    ax.axvline(x=2, color=colours["flapping"], linestyle="--", alpha=0.6,
               label="k = 2 (two gaits)")
    ax.axvline(x=best_k, color=colours["gliding"], linestyle="--", alpha=0.6,
               label=f"BIC minimum (k = {best_k})")
    ax.set_xlabel("Number of GMM clusters (k)")
    ax.set_ylabel("BIC")
    ax.set_title(
        f"GMM model selection ({n_components} PCs, {n_samples:,}-frame subsample)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_continuum_summary(
    scores: np.ndarray,
    flap_idx: np.ndarray,
    glide_idx: np.ndarray,
    flap_kde_idx: np.ndarray,
    glide_kde_idx: np.ndarray,
    traces: Mapping[str, Any],
    flight_phase_colours: Mapping[str, str],
    k_values: Sequence[int],
    bic_curve: np.ndarray,
    *,
    best_k: int,
    colours: Mapping[str, str],
) -> Figure:
    """Three-panel composite summary figure for the flight-behaviour continuum analysis.

    Panel (a) shows the PC1-PC2 morphospace with flapping and gliding overlaid;
    panel (b) shows within-bin PC1 spread versus distance as a wingbeat-activity
    proxy; panel (c) shows the GMM BIC sweep. Together these support the argument
    that flapping and gliding lie on a continuum rather than forming discrete clusters.

    Args:
        scores: PCA score array of shape (n_frames, n_components).
        flap_idx: Row indices of flapping frames in scores.
        glide_idx: Row indices of gliding frames in scores.
        flap_kde_idx: Subset of flap_idx used for KDE estimation.
        glide_kde_idx: Subset of glide_idx used for KDE estimation.
        traces: Dict mapping flight phase name to a binned DataFrame (or None),
            as returned by compute_flight_phase_traces().
        flight_phase_colours: Dict mapping phase name to a hex colour string.
        k_values: Sequence of k values used in the GMM BIC sweep.
        bic_curve: BIC score for each k, aligned with k_values.
        best_k: Value of k at which BIC is minimised.
        colours: Dict with keys 'flapping' and 'gliding' mapping to hex colour
            strings.

    Returns:
        Figure containing the three-panel composite summary.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) PC1-PC2 overlay
    ax = axes[0]
    _scatter_flight_phase(ax, scores, flap_idx, colours["flapping"],
                   alpha=0.12, label="Flapping")
    _scatter_flight_phase(ax, scores, glide_idx, colours["gliding"],
                   alpha=0.12, label="Gliding")
    _draw_kde_contours(ax, scores[flap_kde_idx, :2].T, colours["flapping"])
    _draw_kde_contours(ax, scores[glide_kde_idx, :2].T, colours["gliding"])
    ax.set_xlabel("PC1 (wings down / up)")
    ax.set_ylabel("PC2 (folded / spread)")
    ax.set_title("(a) Gliding: a concentrated subset of flapping")
    ax.legend(loc="lower right", fontsize=9, markerscale=5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")

    # (b) Wingbeat activity = within-bin PC1 std vs distance
    ax = axes[1]
    for name, g in traces.items():
        if g is None:
            continue
        ax.plot(g.index, g["PC1_std"], color=flight_phase_colours[name],
                linewidth=1.8, label=name)
    ax.set_xlabel("Distance to perch (m)")
    ax.set_ylabel("Within-bin std of PC1")
    ax.set_title("(b) Wingbeat activity fades smoothly")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()

    # (c) BIC sweep
    ax = axes[2]
    ax.plot(list(k_values), bic_curve, "ko-", markersize=3)
    ax.axvline(x=best_k, color=colours["gliding"], linestyle="--", alpha=0.6,
               label=f"BIC minimum (k = {best_k})")
    ax.axvspan(2, 3, color=colours["flapping"], alpha=0.15, label="k = 2-3 elbow")
    ax.set_xlabel("Number of GMM clusters (k)")
    ax.set_ylabel("BIC")
    ax.set_title("(c) No discrete small-k solution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
