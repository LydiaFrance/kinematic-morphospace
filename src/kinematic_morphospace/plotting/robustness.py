"""Composite figures for robustness validation supplementary materials.

Each function produces a single figure with the schematic stacked on top
and the corresponding CEV results underneath, at a shared width.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .schematics import (
    _layout_shuffle_schematic, _layout_subsampling_schematic,
    _layout_relabelling_schematic, _layout_imputation_schematic,
    _layout_pairwise_distance_schematic,
)


# ── Shared styling ────────────────────────────────────────────────────

_TITLE_SIZE = 9
_LABEL_SIZE = 8
_TICK_SIZE = 7
_LEGEND_SIZE = 7
_ANNOT_SIZE = 6
_MARKER_SIZE = 3

# Colour scheme
OBSERVED_COLOUR = "#51B3D4"
NULL_COLOUR = "0.55"
NULL_CI_COLOUR = "0.7"
LABEL_COLOUR = "0.4"
TITLE_COLOUR = "0.25"

# Pairwise distance variant colours
LABELLED_COLOUR = "#6ED8A9"
SORTED_COLOUR = "#BC96C9"


def _style_result_ax(ax, title, show_ylabel=False, title_color=LABEL_COLOUR,
                      title_y=0.73, orig_y=1.0):
    """Apply clean styling to a results axes matching schematic aesthetic."""
    # Remove box frame, keep only bottom and left spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.6")
    ax.spines["bottom"].set_color("0.6")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    ax.text(0.5, title_y, title, transform=ax.transAxes, ha="center",
            va="center", fontsize=_TITLE_SIZE, fontweight="bold",
            color=title_color)
    ax.text(0.5, orig_y, "Original data", transform=ax.transAxes, ha="center",
            va="center", fontsize=_TITLE_SIZE, fontweight="bold",
            color=OBSERVED_COLOUR)
    ax.set_xlabel("Number of modes (k)", fontsize=_LABEL_SIZE, color=LABEL_COLOUR)
    ax.tick_params(labelsize=_TICK_SIZE, colors=LABEL_COLOUR, width=0.5)
    if show_ylabel:
        ax.set_ylabel("Cumulative explained variance",
                       fontsize=_LABEL_SIZE, color=LABEL_COLOUR)
    ax.legend(fontsize=_LEGEND_SIZE, loc="lower right", frameon=False)


def _place_results_under_schematics(fig, schem_axes, result_indices,
                                     gap=0.04, result_frac=0.75,
                                     width_scale=1.0):
    """Create result axes positioned directly under specific schematic axes.

    Parameters
    ----------
    fig : Figure
    schem_axes : list of Axes
        All schematic axes (top row).
    result_indices : list of int
        Which schematic axes to place results under.
    gap : float
        Gap between schematic bottom and result top (figure coords).
    result_frac : float
        Result height as fraction of the schematic height.
    width_scale : float
        Scale factor for result width relative to schematic width.
        Values < 1.0 narrow the result axes (centred under schematic).

    Returns
    -------
    result_axes : list of Axes
    """
    # Force a draw so axes positions are resolved (needed for aspect="equal")
    fig.canvas.draw()

    # Find the lowest schematic bottom edge across the relevant axes
    schem_bottom = min(schem_axes[i].get_position().y0
                       for i in result_indices)
    # Derive result dimensions from actual schematic size
    sample_pos = schem_axes[result_indices[0]].get_position()
    result_height = sample_pos.height * result_frac
    result_top = schem_bottom - gap
    result_bottom = result_top - result_height

    result_axes = []
    for i in result_indices:
        pos = schem_axes[i].get_position()
        w = pos.width * width_scale
        x0 = pos.x0 + (pos.width - w) / 2  # centre under schematic
        ax = fig.add_axes([x0, result_bottom, w, result_height])
        result_axes.append(ax)
    return result_axes


# ── Results layout helpers ────────────────────────────────────────────

def _layout_shuffle_results(axes, cev, shuffle_results, n_comp):
    """Draw shuffle CEV results onto 4 axes."""
    mode_names = ["temporal", "column", "label", "complete"]
    mode_labels = ["Temporal shuffle", "Column shuffle",
                   "Label shuffle", "Complete shuffle"]
    k_vals = np.arange(1, n_comp + 1)

    for ax, mode, label in zip(axes, mode_names, mode_labels):
        null_cev = shuffle_results[mode]
        null_mean = np.mean(null_cev, axis=0)
        null_lo = np.percentile(null_cev, 2.5, axis=0)
        null_hi = np.percentile(null_cev, 97.5, axis=0)

        ax.fill_between(k_vals, null_lo, null_hi, alpha=0.2, color=NULL_CI_COLOUR,
                         label='Null 95% CI')
        ax.plot(k_vals, null_mean, 'o--', color=NULL_COLOUR, label='Null mean',
                markersize=_MARKER_SIZE - 1)
        ax.plot(k_vals, cev, 'o-', color='#51B3D4', label='Observed',
                markersize=_MARKER_SIZE)

        ax.set_xlim(1, 12)
        ax.set_ylim(0, 1.05)
        _style_result_ax(ax, label, show_ylabel=(ax is axes[0]))


def _layout_subsampling_results(axes, cev, results_subsets):
    """Draw subsampling CEV results onto 4 axes."""
    k_full = np.arange(1, len(cev) + 1)

    for ax, result in zip(axes, results_subsets):
        held_out = result["held_out"]
        subset_cev = result["cev"]
        k_sub = np.arange(1, len(subset_cev) + 1)

        ax.plot(k_full, cev, 'o-', color='#51B3D4', label='Full (4 markers)',
                markersize=_MARKER_SIZE)
        ax.plot(k_sub, subset_cev, 'o--', color=NULL_COLOUR,
                label=f'Drop {held_out}', markersize=_MARKER_SIZE)

        cosines = result["cosines"]
        cos_str = ", ".join(f"{c:.2f}" for c in cosines)
        ax.text(0.97, 0.4, f"cosines: {cos_str}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=_ANNOT_SIZE, color=LABEL_COLOUR)

        ax.set_xlim(1, 9)
        ax.set_ylim(0, 1.05)
        _style_result_ax(ax, f"Drop {held_out}",
                         show_ylabel=(ax is axes[0]),
                         title_color=TITLE_COLOUR, title_y=0.8, orig_y=1.0)


def _layout_pairwise_results(axes, cev, pw_cev, pw_sorted_cev,
                              pw_shuffled_cev):
    """Draw pairwise distance CEV results onto 3 axes."""
    k_vals = np.arange(1, 7)
    variant_data = [
        ("Labelled distances", pw_cev, LABELLED_COLOUR, 's-'),
        ("Sorted distances", pw_sorted_cev, SORTED_COLOUR, '^-'),
        ("Shuffled distances\n(within-frame)", pw_shuffled_cev, 'grey', 'x--'),
    ]

    for ax, (label, var_cev, colour, fmt) in zip(axes, variant_data):
        ax.plot(k_vals, cev[:6], 'o-', color='#51B3D4',
                label='Marker coords', markersize=_MARKER_SIZE)
        ax.plot(k_vals, var_cev[:6], fmt, color=colour, label=label,
                markersize=_MARKER_SIZE + 1)

        ax.set_xlim(1, 6)
        ax.set_ylim(0, 1.05)
        _style_result_ax(ax, label, show_ylabel=(ax is axes[0]))


def _layout_relabelling_results(axes, cev, relabel_results, fractions,
                                 n_comp):
    """Draw relabelling CEV results onto axes (one per fraction)."""
    k_vals = np.arange(1, n_comp + 1)

    for ax, frac in zip(axes, fractions):
        cev_dist = relabel_results[frac]["cev"]
        cev_mean = np.mean(cev_dist, axis=0)
        cev_lo = np.percentile(cev_dist, 2.5, axis=0)
        cev_hi = np.percentile(cev_dist, 97.5, axis=0)

        ax.fill_between(k_vals, cev_lo, cev_hi, alpha=0.2, color=NULL_CI_COLOUR,
                         label='Relabelled 95% CI')
        ax.plot(k_vals, cev_mean, 'o--', color=NULL_COLOUR,
                label='Relabelled mean', markersize=_MARKER_SIZE - 1)
        ax.plot(k_vals, cev, 'o-', color='#51B3D4', label='Observed',
                markersize=_MARKER_SIZE)

        cos_mean = np.mean(relabel_results[frac]["cosines"], axis=0)
        cos_str = ", ".join(f"{c:.2f}" for c in cos_mean)
        ax.text(0.97, 0.4, f"cosines: {cos_str}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=_ANNOT_SIZE, color=LABEL_COLOUR)

        ax.set_xlim(1, 12)
        ax.set_ylim(0, 1.05)
        _style_result_ax(ax, f"{frac:.0%} relabelling",
                         show_ylabel=(ax is axes[0]))


def _layout_imputation_results(ax, cev, imputed_cev, cosines, n_comp):
    """Draw imputation CEV results onto a single axes."""
    k_vals = np.arange(1, n_comp + 1)

    ax.plot(k_vals, cev, 'o-', color='#51B3D4',
            label='Original (no missing)', markersize=_MARKER_SIZE)
    ax.plot(k_vals, imputed_cev, 'o--', color=NULL_COLOUR,
            label='Imputed (468k frames)', markersize=_MARKER_SIZE)

    cos_str = ", ".join(f"{cosines[m]:.2f}" for m in range(len(cosines)))
    ax.text(0.97, 0.4, f"cosines: {cos_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=_ANNOT_SIZE, color="0.3")

    ax.set_xlim(1, 12)
    ax.set_ylim(0, 1.05)
    _style_result_ax(ax, "Missing data imputation", show_ylabel=True)


# ── Composite figures ─────────────────────────────────────────────────

def plot_shuffle_composite(cev, shuffle_results, n_comp=12, fig_width=20):
    """Schematic + results for shuffle controls."""
    fig = plt.figure(figsize=(fig_width, 10))

    # Top row: schematics (5 panels, equal aspect)
    gs_top = GridSpec(1, 5, figure=fig, left=0.03, right=0.97,
                       top=0.95, bottom=0.40, wspace=0.32)
    schem_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
    _layout_shuffle_schematic(schem_axes)

    # Place results directly under schematic columns 1–4
    result_axes = _place_results_under_schematics(
        fig, schem_axes, [1, 2, 3, 4],
        gap=0.04, result_frac=0.75, width_scale=0.8)
    _layout_shuffle_results(result_axes, cev, shuffle_results, n_comp)

    return fig


def plot_subsampling_composite(cev, results_subsets, fig_width=20):
    """Schematic + results for subsampling controls."""
    fig = plt.figure(figsize=(fig_width, 10))

    gs_top = GridSpec(1, 5, figure=fig, left=0.03, right=0.97,
                       top=0.95, bottom=0.40, wspace=0.1)
    schem_axes = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
    _layout_subsampling_schematic(schem_axes)

    result_axes = _place_results_under_schematics(
        fig, schem_axes, [1, 2, 3, 4],
        gap=0.04, result_frac=0.75, width_scale=0.6)
    _layout_subsampling_results(result_axes, cev, results_subsets)

    return fig


def plot_pairwise_composite(cev, pw_cev, pw_sorted_cev, pw_shuffled_cev,
                             fig_width=14):
    """Schematic + results for pairwise distance controls."""
    fig = plt.figure(figsize=(fig_width, 9))

    # Pairwise schematic uses its own internal GridSpec
    gs_top = GridSpec(1, 1, figure=fig, left=0.03, right=0.97,
                       top=0.95, bottom=0.30)
    _layout_pairwise_distance_schematic(fig, gs_top[0, 0])

    # For pairwise, we need to find the 3 distance panel axes (not the
    # marker-coords or explainer axes). They're the last 3 axes added.
    all_axes = fig.get_axes()
    # The distance panels are axes at indices 2, 3, 4 (after ax0 and ax_exp)
    dist_axes = all_axes[2:5]

    result_axes = _place_results_under_schematics(
        fig, dist_axes, [0, 1, 2],
        gap=0.02, result_frac=0.45, width_scale=0.9)
    _layout_pairwise_results(result_axes, cev, pw_cev, pw_sorted_cev,
                              pw_shuffled_cev)

    return fig


def plot_relabelling_composite(cev, relabel_results, fractions=(0.05, 0.25),
                                n_comp=12, fig_width=12):
    """Schematic + results for relabelling controls."""
    n_panels = 1 + len(fractions)
    fig = plt.figure(figsize=(fig_width, 10))

    gs_top = GridSpec(1, n_panels, figure=fig, left=0.05, right=0.95,
                       top=0.95, bottom=0.30, wspace=0.35)
    schem_axes = [fig.add_subplot(gs_top[0, i]) for i in range(n_panels)]
    _layout_relabelling_schematic(schem_axes, fractions=fractions)

    result_axes = _place_results_under_schematics(
        fig, schem_axes, list(range(1, n_panels)),
        gap=0.06, result_frac=0.55)
    _layout_relabelling_results(result_axes, cev, relabel_results, fractions,
                                 n_comp)

    return fig


def plot_imputation_composite(cev, imputed_cev, cosines, n_comp=12,
                               fig_width=12):
    """Schematic + results for imputation controls."""
    fig = plt.figure(figsize=(fig_width, 10))

    gs_top = GridSpec(1, 3, figure=fig, left=0.05, right=0.95,
                       top=0.95, bottom=0.30, wspace=0.35)
    schem_axes = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    _layout_imputation_schematic(schem_axes)

    # Place result centred between columns 1 (Missing) and 2 (Imputed)
    fig.canvas.draw()
    pos1 = schem_axes[1].get_position()
    pos2 = schem_axes[2].get_position()
    schem_bottom = min(pos1.y0, pos2.y0)
    result_height = pos1.height * 0.55
    result_top = schem_bottom - 0.06
    result_bottom = result_top - result_height
    # Centre a square-ish result between cols 1 and 2
    mid_x = (pos1.x0 + pos1.width + pos2.x0) / 2
    w = result_height * (fig.get_figheight() / fig.get_figwidth())  # square
    result_ax = fig.add_axes([mid_x - w / 2, result_bottom, w, result_height])
    _layout_imputation_results(result_ax, cev, imputed_cev, cosines, n_comp)

    return fig


def plot_hull_coverage(pts_labelled, pts_unlabelled, pca_embed,
                       coverage_rev=None, figsize=(7, 6)):
    """Scatter + marginals showing labelled vs unlabelled frame overlap.

    Central panel shows both groups as translucent scatter points.
    Top and right margins show normalised density histograms for PC1 and PC2.
    Coverage statistics are annotated directly on the plot.

    Parameters
    ----------
    pts_labelled : ndarray, shape (n_labelled, n_components)
        PCA-embedded pairwise-distance histograms for labelled frames.
    pts_unlabelled : ndarray, shape (n_unlabelled, n_components)
        PCA-embedded pairwise-distance histograms for unlabelled frames.
    pca_embed : PCA
        Fitted PCA used for embedding (for axis labels).
    coverage_rev : float, optional
        Fraction of unlabelled frames inside the labelled convex hull.

    Returns
    -------
    fig : Figure
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[4, 1], height_ratios=[1, 4],
                  hspace=0.04, wspace=0.04)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Colours
    c_unlab = "mediumorchid" \
    ""
    c_lab = OBSERVED_COLOUR

    # --- Central scatter ---
    ax_main.scatter(
        pts_unlabelled[:, 0], pts_unlabelled[:, 1],
        s=6, alpha=0.15, color=c_unlab, label="Unlabelled", rasterized=True,
    )
    ax_main.scatter(
        pts_labelled[:, 0], pts_labelled[:, 1],
        s=6, alpha=0.25, color=c_lab, label="Labelled", rasterized=True,
    )

    # --- Marginal histograms ---
    bins_x = np.linspace(
        min(pts_unlabelled[:, 0].min(), pts_labelled[:, 0].min()),
        max(pts_unlabelled[:, 0].max(), pts_labelled[:, 0].max()),
        40,
    )
    bins_y = np.linspace(
        min(pts_unlabelled[:, 1].min(), pts_labelled[:, 1].min()),
        max(pts_unlabelled[:, 1].max(), pts_labelled[:, 1].max()),
        40,
    )

    ax_top.hist(pts_unlabelled[:, 0], bins=bins_x, density=True,
                alpha=0.4, color=c_unlab, edgecolor="none")
    ax_top.hist(pts_labelled[:, 0], bins=bins_x, density=True,
                alpha=0.4, color=c_lab, edgecolor="none")

    ax_right.hist(pts_unlabelled[:, 1], bins=bins_y, density=True,
                  orientation="horizontal", alpha=0.4, color=c_unlab,
                  edgecolor="none")
    ax_right.hist(pts_labelled[:, 1], bins=bins_y, density=True,
                  orientation="horizontal", alpha=0.4, color=c_lab,
                  edgecolor="none")

    # --- Coverage annotation ---
    if coverage_rev is not None:
        ax_main.text(
            0.03, 0.97,
            f"Unlabelled shapes inside labelled hull: {coverage_rev:.0%}",
            transform=ax_main.transAxes,
            va="top", ha="left", fontsize=_LABEL_SIZE, color=LABEL_COLOUR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8",
                      alpha=0.85),
        )

    # --- Labels and styling ---
    var = pca_embed.explained_variance_ratio_
    ax_main.set_xlabel(f"PC1 ({var[0]:.1%} var.)", fontsize=_LABEL_SIZE,
                       color=LABEL_COLOUR)
    ax_main.set_ylabel(f"PC2 ({var[1]:.1%} var.)", fontsize=_LABEL_SIZE,
                       color=LABEL_COLOUR)
    ax_main.legend(frameon=False, fontsize=_LEGEND_SIZE, loc="lower right")

    for ax in [ax_main, ax_top, ax_right]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("0.6")
        ax.spines["bottom"].set_color("0.6")
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(labelsize=_TICK_SIZE, colors=LABEL_COLOUR, width=0.5)

    # Hide tick labels on marginals
    ax_top.tick_params(labelbottom=False)
    ax_top.set_yticks([])
    ax_right.tick_params(labelleft=False)
    ax_right.set_xticks([])

    # Corner cell (top-right) — hide it
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")

    fig.tight_layout()
    return fig


def plot_hull_outlier_markers(marker_counts, outside_mask, figsize=(5, 3.5)):
    """Outlier rate vs marker count, showing the confound.

    Each dot is one marker-count value (2, 3, ... 20+). The y-axis shows
    what percentage of frames with that count fall outside the labelled
    convex hull. Dot size encodes sample size. A vertical line at 8
    marks the expected feather marker count.

    Parameters
    ----------
    marker_counts : ndarray of int
        Number of visible marker detections per unlabelled frame.
    outside_mask : ndarray of bool
        True for frames outside the labelled convex hull.

    Returns
    -------
    fig : Figure
    """
    unique_counts = np.unique(marker_counts)
    outlier_rates = []
    sample_sizes = []

    for c in unique_counts:
        mask = marker_counts == c
        n = mask.sum()
        sample_sizes.append(n)
        outlier_rates.append(outside_mask[mask].mean() * 100)

    outlier_rates = np.array(outlier_rates)
    sample_sizes = np.array(sample_sizes)

    # Scale dot sizes: smallest group gets min_s, largest gets max_s
    min_s, max_s = 20, 200
    if sample_sizes.max() > sample_sizes.min():
        scaled = (sample_sizes - sample_sizes.min()) / (
            sample_sizes.max() - sample_sizes.min())
        sizes = min_s + scaled * (max_s - min_s)
    else:
        sizes = np.full_like(sample_sizes, (min_s + max_s) / 2, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(unique_counts, outlier_rates, s=sizes, color="mediumorchid",
               alpha=0.7, edgecolor="white", linewidth=0.5, zorder=3)

    # Reference line at 8 markers
    ax.axvline(8, color=OBSERVED_COLOUR, ls="--", lw=1, alpha=0.6,
               label="Expected (8 feather markers)")

    # Overall outlier rate
    overall = outside_mask.mean() * 100
    ax.axhline(overall, color="0.7", ls=":", lw=0.8)
    ax.text(unique_counts.max() + 0.3, overall, f"overall\n{overall:.0f}%",
            va="center", ha="left", fontsize=_ANNOT_SIZE, color=LABEL_COLOUR)

    ax.set_xlabel("Marker detections per frame", fontsize=_LABEL_SIZE,
                  color=LABEL_COLOUR)
    ax.set_ylabel("Frames outside hull (%)", fontsize=_LABEL_SIZE,
                  color=LABEL_COLOUR)
    ax.legend(frameon=False, fontsize=_LEGEND_SIZE, loc="upper right")

    ax.set_xlim(1, unique_counts.max() + 1.5)
    ax.set_ylim(0, None)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.6")
    ax.spines["bottom"].set_color("0.6")
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(labelsize=_TICK_SIZE, colors=LABEL_COLOUR, width=0.5)

    fig.tight_layout()
    return fig


def plot_occlusion_bias(complete_scores, partial_scores, labels,
                        n_bins=30, figsize=(10, 3.5)):
    """Overlaid histograms of PC scores for complete vs partial frames.

    Two-panel figure: one per morphing axis. Each panel shows normalised
    density histograms for complete-marker frames and partial-marker
    frames (scored via least-squares projection). Differences reveal
    which wing shapes are underrepresented due to marker occlusion.

    Parameters
    ----------
    complete_scores : dict
        Keys are axis labels. Values are 1D arrays of scores for
        complete frames.
    partial_scores : dict
        Same keys. Values are 1D arrays of scores for partial frames.
    labels : tuple of str
        Legend labels, e.g. ("Complete (8 markers)", "Partial (<8)").
    n_bins : int
        Number of equal-width histogram bins.

    Returns
    -------
    fig : Figure
    """
    n_panels = len(complete_scores)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    c_complete = OBSERVED_COLOUR
    c_partial = "mediumorchid"

    for ax, key in zip(axes, complete_scores):
        comp = complete_scores[key]
        part = partial_scores[key]

        # Shared bin edges
        lo = min(np.percentile(comp, 1), np.percentile(part, 1))
        hi = max(np.percentile(comp, 99), np.percentile(part, 99))
        bins = np.linspace(lo, hi, n_bins + 1)

        ax.hist(comp, bins=bins, density=True, alpha=0.5,
                color=c_complete, edgecolor="none", label=labels[0])
        ax.hist(part, bins=bins, density=True, alpha=0.5,
                color=c_partial, edgecolor="none", label=labels[1])

        ax.set_xlabel(key, fontsize=_LABEL_SIZE, color=LABEL_COLOUR)

        if "(" in key:
            title = key.split("(")[1].rstrip(")")
            title = title[0].upper() + title[1:]
        else:
            title = key
        ax.set_title(title, fontsize=_TITLE_SIZE, color=TITLE_COLOUR)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("0.6")
        ax.spines["bottom"].set_color("0.6")
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(labelsize=_TICK_SIZE, colors=LABEL_COLOUR, width=0.5)

    axes[0].set_ylabel("Density", fontsize=_LABEL_SIZE, color=LABEL_COLOUR)
    axes[-1].legend(frameon=False, fontsize=_LEGEND_SIZE)
    fig.tight_layout()
    return fig
