"""Left-right symmetry visualisation for bilateral PC scores."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from ..data_filtering import filter_by
from .markers import plot_raw_markers


def prepare_left_right_comparison(scores_df, *, scores_df_norot=None, **filters):
    """Prepare merged left/right PC scores for bilateral symmetry analysis.

    Filters left- and right-wing scores separately, merges them on frameID
    into a single row per frame, and computes axis-limit percentiles for
    consistent cross-panel scaling. Manual overrides for specific PCs ensure
    that all panels share a visually consistent scale.

    Args:
        scores_df: DataFrame containing PC scores and metadata; must include
            frameID, seqID, and left columns.
        scores_df_norot: Optional DataFrame of unrotated scores. If provided,
            PC01 and PC02 columns are taken from this dataset (avoiding the
            circularity of rotation correction which assumes symmetry for these
            modes), while PC03+ come from scores_df. Both DataFrames must share
            the same frameIDs.
        **filters: Keyword arguments forwarded to filter_by() to select the
            subset of flights to include (e.g. obstacle=0, hawkname='Drogon').

    Returns:
        Tuple of (left_right_scores, score_5, score_95) where left_right_scores
        is a merged DataFrame with _left and _right suffixed PC columns, and
        score_5 / score_95 are per-PC lower and upper axis-limit Series.
    """
    # Create base filters
    left_filter = filter_by(scores_df, left=1, **filters)
    right_filter = filter_by(scores_df, left=0, **filters)


    # Get left and right scores
    left_scores = scores_df[left_filter].set_index('frameID')
    right_scores = scores_df[right_filter].set_index('frameID')

    # Merge left and right scores
    left_right_scores = left_scores.merge(
        right_scores,
        left_index=True,
        right_index=True,
        suffixes=('_left', '_right'),
    )

    # If unrotated scores provided, swap in PC01 and PC02 from that dataset
    # (avoids circularity: rotation correction assumes symmetry for modes 1-2)
    if scores_df_norot is not None:
        left_filter_norot = filter_by(scores_df_norot, left=1, **filters)
        right_filter_norot = filter_by(scores_df_norot, left=0, **filters)
        left_norot = scores_df_norot[left_filter_norot].set_index('frameID')
        right_norot = scores_df_norot[right_filter_norot].set_index('frameID')
        lr_norot = left_norot.merge(
            right_norot, left_index=True, right_index=True,
            suffixes=('_left', '_right'),
        )
        # Replace PC01 and PC02 columns with unrotated versions
        for pc in ['PC01', 'PC02']:
            left_right_scores[f'{pc}_left'] = lr_norot[f'{pc}_left']
            left_right_scores[f'{pc}_right'] = lr_norot[f'{pc}_right']

    # Calculate score percentiles
    PC_cols = [f'PC{i:02}' for i in np.arange(1, 13)]
    flying_filter = filter_by(scores_df, horzdist='in-flight')
    score_95 = scores_df.loc[flying_filter, PC_cols].quantile(0.998)
    score_5 = scores_df.loc[flying_filter, PC_cols].quantile(0.002)

    # Manual axis limits override the data-driven percentiles for
    # publication figures.  These are rounded, symmetric bounds chosen
    # so that all panels share a visually consistent scale and the
    # data-driven 0.2/99.8 percentiles do not produce jagged limits.
    # To use purely data-driven limits, delete or empty this dict.
    pc_limits = {
        'PC01': (0.6, -0.6),
        'PC02': (0.45, -0.45),
        'PC03': (0.2, -0.2),
        'PC04': (0.15, -0.15),
        'PC06': (0.09, -0.09),
        'PC07': (0.1, -0.1),
        'PC09': (0.07, -0.07),
        'PC11': (0.05, -0.05),
        'PC12': (0.04, -0.04)
    }

    for pc, (high, low) in pc_limits.items():
        score_95[pc] = high
        score_5[pc] = low

    print(f"Number of frames: {len(left_right_scores)}")
    print(f"Number of flights: {len(np.unique(left_scores['seqID']))}")

    return left_right_scores, score_5, score_95


# ---------- Shared helpers for left-right symmetry plots ----------

def _major_axis_regression(data):
    """Fit a PCA major-axis regression line to 2-D bivariate data.

    This is major-axis (MA) regression — the first eigenvector of the bivariate
    cloud — not reduced major-axis (RMA) regression.

    Args:
        data: Array of shape (N, 2) containing paired left/right scores.

    Returns:
        Tuple of (slope, intercept, variance_pct) where slope and intercept
        define the major-axis line and variance_pct is the percentage of
        variance explained by the first principal component.
    """
    pca = PCA(n_components=2)
    pca.fit(data)
    major_axis = pca.components_[0]

    mean_x, mean_y = np.mean(data, axis=0)

    major_slope = major_axis[1] / major_axis[0]
    major_intercept = mean_y - major_slope * mean_x

    percentage_variance = pca.explained_variance_ratio_[0] * 100

    return major_slope, major_intercept, percentage_variance


def _major_axis_line(data, major_slope):
    mean_x, mean_y = np.mean(data, axis=0)
    min_val = np.min(data)
    max_val = np.max(data)

    x_grid = np.linspace(min_val, max_val, 100)
    major_line = mean_y + major_slope * (x_grid - mean_x)

    return x_grid, major_line


def _plot_one_pc(ax, left_right_scores, score_5, score_95, PC,
                 colour, alpha, bkgrd_color='white', point_size=0.1):
    """Render a single left-vs-right scatter panel for one PC."""
    pc_label = f'PC{PC+1:02}'

    data = np.array([left_right_scores[f'{pc_label}_right'],
                     left_right_scores[f'{pc_label}_left']]).T
    major_slope, major_intercept, percentage_variance = _major_axis_regression(data)
    x_grid, major_line = _major_axis_line(data, major_slope)

    plot_raw_markers(ax,
                     left_right_scores[f'{pc_label}_right'],
                     left_right_scores[f'{pc_label}_left'],
                     colour=colour, alpha=alpha, size=point_size)

    ax.plot(x_grid, major_line, ':', c='black', linewidth=0.8)
    ax.set_facecolor(bkgrd_color)

    min_val = score_5[pc_label]
    max_val = score_95[pc_label]
    ax.plot([min_val, max_val], [min_val, max_val], '-', c='grey', linewidth=0.8)

    ax.set_xlim(min_val, max_val)
    ax.set_xticks([min_val, 0, max_val])
    ax.set_xticklabels([f'{min_val:.2f}', '0', f'{max_val:.2f}'], fontsize=6)
    ax.set_ylim(min_val, max_val)
    ax.set_yticks([0, max_val])
    ax.set_yticklabels(['0', f'{max_val:.2f}'], fontsize=6)
    ax.set_ylabel(f'PC{PC+1}', fontsize=8)
    ax.grid(True)

    if major_intercept == 0:
        ax.text(0.05, 0.9, f'y = {major_slope:.2f}x',
                transform=ax.transAxes, fontsize=8)
    else:
        plusminus = '+' if major_intercept > 0 else '-'
        ax.text(0.052, 0.9,
                f'y = {major_slope:.2f}x {plusminus} {abs(major_intercept):.3f}',
                transform=ax.transAxes, fontsize=8)
        ax.text(0.05, 0.8, f'{percentage_variance:.1f}%',
                transform=ax.transAxes, fontsize=8)


# ---------- Public plotting functions ----------

def summarise_symmetry(left_right_scores):
    """Compute major-axis regression statistics for all 12 PC modes.

    For each PC, computes the slope, intercept, and variance explained by the
    major-axis regression of left scores against right scores. Qualitative
    coupling labels (strong / moderate / weak) are assigned based on how close
    the slope is to 1.0 (perfect symmetry).

    Args:
        left_right_scores: Merged left/right DataFrame as returned by
            prepare_left_right_comparison().

    Returns:
        DataFrame with one row per PC and columns: mode, slope, intercept,
        variance_pct, and coupling (qualitative symmetry label).
    """
    rows = []
    for pc in range(12):
        pc_label = f'PC{pc + 1:02}'
        data = np.array([left_right_scores[f'{pc_label}_right'],
                         left_right_scores[f'{pc_label}_left']]).T
        slope, intercept, var_pct = _major_axis_regression(data)
        # Qualitative coupling label based on slope proximity to 1.0
        dev = abs(slope - 1.0)
        if dev < 0.05:
            coupling = 'strong'
        elif dev < 0.15:
            coupling = 'moderate'
        else:
            coupling = 'weak'
        rows.append({
            'mode': pc_label,
            'slope': slope,
            'intercept': intercept,
            'variance_pct': var_pct,
            'coupling': coupling,
        })
    return pd.DataFrame(rows)


def print_symmetry_summary(left_right_scores, label=''):
    """Print a formatted major-axis regression summary for all 12 PC modes.

    Args:
        left_right_scores: Merged left/right DataFrame as returned by
            prepare_left_right_comparison().
        label: Optional heading string printed before the table (e.g.
            'Flapping'). If empty, no heading is printed. Defaults to ''.
    """
    df = summarise_symmetry(left_right_scores)
    if label:
        print(f'\n--- {label} ---')
    print(f'  {"Mode":<6} {"Slope":>7} {"Intercept":>10} {"Var %":>7}  {"Coupling"}')
    for _, r in df.iterrows():
        print(f'  {r["mode"]:<6} {r["slope"]:>7.3f} {r["intercept"]:>10.4f} '
              f'{r["variance_pct"]:>6.1f}%  {r["coupling"]}')


DEFAULT_PC_ORDER = [0, 1, 2, 3, 4, 7, 5, 6, 8, 9, 10, 11]
"""Thematic display order for morphing modes: PC01-05, then PC08 (collective
pitching, pitch-related and grouped with PC05 counter-pitching), then the
handwing trio PC06, PC07, PC09, then PC10-12."""


def plot_left_right(
    left_right_scores, score_5, score_95, alpha=0.05, bkgrd_color='white',
    highlight_unrotated=False, pc_order=[0, 1, 2, 3, 4, 7, 5, 6, 8], point_size=0.1,
):
    """Plot grid comparing left vs right PC scores.

    Each panel shows a scatter of left-wing scores against right-wing scores
    for one PC, with the major-axis regression line and the line of perfect
    symmetry (y = x) overlaid. Deviation from the diagonal reveals left-right
    asymmetry in that morphing mode.

    Args:
        left_right_scores: Merged left/right DataFrame as returned by
            prepare_left_right_comparison().
        score_5: Per-PC lower axis limits.
        score_95: Per-PC upper axis limits.
        alpha: Scatter-point opacity. Defaults to 0.05.
        bkgrd_color: Background colour for each panel. Defaults to 'white'.
        highlight_unrotated: If True, draw a dotted box around modes 1-2 to
            indicate these panels use unrotated (pre-correction) scores.
            Defaults to False.
        pc_order: Zero-indexed PC numbers to plot, in display order. Defaults
            to DEFAULT_PC_ORDER (PC01-05, PC08, PC06, PC07, PC09-12). Pass a
            9-element list to drop PC10-12 from conditional figures where
            those higher modes are interpreted as noise.
        point_size: Scatter marker size passed to matplotlib. Defaults to 0.1
            (suitable for dense all-flight plots). Increase for sparse subsets
            such as obstacle flights where individual points are hard to see.

    Returns:
        Tuple of (fig, axs) where axs is a flat array of Axes.
    """
    if pc_order is None:
        pc_order = list(DEFAULT_PC_ORDER)
    n = len(pc_order)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(8, 2 * nrows),
                            sharex=False, sharey=False,
                            gridspec_kw={'hspace': 0.15, 'wspace': 0})
    axs = axs.flatten()
    colour_list = ['#B5E675', '#6ED8A9', '#51B3D4',
              '#4579AA', '#F19EBA', '#BC96C9',
              '#917AC2', '#BE607F', '#624E8B',
              '#888888', '#888888', '#888888']

    for i, PC in enumerate(pc_order):
        _plot_one_pc(axs[i], left_right_scores, score_5, score_95,
                     PC, colour_list[PC], alpha, bkgrd_color,
                     point_size=point_size)

    for j in range(n, len(axs)):
        axs[j].axis('off')

    if highlight_unrotated:
        _add_unrotated_highlight(fig, axs)

    return fig, axs


def _add_unrotated_highlight(fig, axs):
    """Draw a dotted box around modes 1-2 to indicate unrotated scores.

    Args:
        fig: The figure containing the axes.
        axs: Flat array of axes from the 4x3 grid.
    """
    from matplotlib.patches import FancyBboxPatch

    # Get bounding box around first two axes (PC1 and PC2)
    bbox0 = axs[0].get_position()
    bbox1 = axs[1].get_position()

    # Compute rectangle that encompasses both, with small padding
    pad = 0.01
    x0 = bbox0.x0 - (pad*4.5)
    y0 = min(bbox0.y0, bbox1.y0) - pad
    width = bbox1.x1 - bbox0.x0 + 2 * (pad*2.5)
    height = max(bbox0.y1, bbox1.y1) - y0 + pad

    rect = FancyBboxPatch(
        (x0, y0), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1,
        edgecolor='#666666',
        facecolor='none',
        linestyle=':',
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(rect)

    # Add small label
    fig.text(
        x0 + width, y0 + height + (pad*2.5),
        'without rotation correction',
        ha='center', va='top',
        fontsize=7, color='#666666',
        transform=fig.transFigure,
    )


def plot_left_right_just_two(left_right_scores, score_5, score_95, alpha=0.05):
    """Plot left-vs-right symmetry panels for PC1 and PC2 only.

    Intended for showing scores before the rotation-correction step, where
    only the first two components have been bilateralised. The remaining 10
    panels are hidden.

    Args:
        left_right_scores: Merged left/right DataFrame as returned by
            prepare_left_right_comparison().
        score_5: Per-PC lower axis limits.
        score_95: Per-PC upper axis limits.
        alpha: Scatter-point opacity. Defaults to 0.05.

    Returns:
        Tuple of (fig, axs) where axs is a flat array of 2 Axes.
    """
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.5),
                            sharex=False, sharey=False,
                            gridspec_kw={'wspace': 0})
    axs = axs.flatten()
    colour_list = ['#B5E675', '#6ED8A9']

    for PC in range(2):
        _plot_one_pc(axs[PC], left_right_scores, score_5, score_95,
                     PC, colour_list[PC], alpha)

    return fig, axs


def plot_left_right_just_two_stacked(conditions, score_5, score_95, alpha=0.05):
    """Plot PC1 and PC2 symmetry panels for multiple conditions, stacked vertically.

    Each row shows one condition (e.g. straight flights, obstacle flights) with
    two panels: PC1 on the left and PC2 on the right. Row labels are added on
    the left margin from the ``title`` key of each condition dict.

    Args:
        conditions: List of dicts, each with keys:
            - ``title``: str label for the row (e.g. ``'Straight flights'``).
            - ``left_right_scores``: DataFrame from
              :func:`prepare_left_right_comparison`.
        score_5: Per-PC lower axis limits (shared across conditions).
        score_95: Per-PC upper axis limits (shared across conditions).
        alpha: Scatter-point opacity, either a single float applied to all
            conditions or a list with one value per condition. Defaults to 0.05.

    Returns:
        Tuple of (fig, axs) where axs has shape (n_conditions, 2).
    """
    n = len(conditions)
    alphas = alpha if isinstance(alpha, (list, tuple)) else [alpha] * n
    colour_list = ['#B5E675', '#6ED8A9']
    fig, axs = plt.subplots(n, 2, figsize=(5.5, 2.5 * n),
                            sharex=False, sharey=False,
                            gridspec_kw={'hspace': 0.35, 'wspace': 0})
    if n == 1:
        axs = axs[np.newaxis, :]

    for row, (condition, a) in enumerate(zip(conditions, alphas)):
        for PC in range(2):
            _plot_one_pc(axs[row, PC], condition['left_right_scores'],
                         score_5, score_95, PC, colour_list[PC], alpha = a)
        axs[row, 0].set_title(condition['title'], fontsize=9, loc='left', pad=4)

    return fig, axs


def plot_rotation_correction_comparison(scores_df, scores_df_norot, alpha=0.5, **filters):
    """Compare corrected vs uncorrected scores for modes 1-2.

    Creates a stacked figure showing the effect of rotation correction on
    bilateral symmetry for the first two modes. The top row shows corrected
    scores (with artificially forced symmetry), the bottom row shows uncorrected
    scores (with genuine bilateral coupling).

    This comparison validates the assumption underlying the rotation correction:
    that modes 1-2 are genuinely bilaterally symmetric.

    Args:
        scores_df: DataFrame of rotation-corrected scores.
        scores_df_norot: DataFrame of uncorrected (pre-rotation) scores.
        alpha: Scatter-point opacity. Defaults to 0.5.
        **filters: Keyword arguments forwarded to prepare_left_right_comparison()
            (e.g. obstacle=0, naive=0, perchDist=["9", "12"]).

    Returns:
        Tuple of (fig, axs) where axs has shape (2, 2).
    """
    # Prepare corrected scores
    lr_corrected, score_5, score_95 = prepare_left_right_comparison(
        scores_df, **filters
    )
    # Prepare uncorrected scores
    lr_uncorrected, _, _ = prepare_left_right_comparison(
        scores_df_norot, **filters
    )

    # Build conditions for stacked plot
    conditions = [
        {
            "title": "Rotation-corrected (modes 1-2 have forced symmetry)",
            "left_right_scores": lr_corrected,
        },
        {
            "title": "Uncorrected (genuine symmetry for modes 1-2)",
            "left_right_scores": lr_uncorrected,
        },
    ]

    fig, axs = plot_left_right_just_two_stacked(
        conditions, score_5, score_95, alpha=[alpha, alpha]
    )

    return fig, axs


def plot_symmetry_validation(scores_df, scores_df_norot, alpha_straight=0.5, alpha_obstacle=0.9):
    """Compare corrected vs uncorrected scores for modes 1-2 across flight types.

    Creates a 4-row stacked figure comparing bilateral symmetry for modes 1-2
    in both straight and obstacle flights, with and without rotation correction.
    This validates the assumption that modes 1-2 are genuinely symmetric even
    during turning flight.

    Rows:
        1. Straight flights — uncorrected (underlying symmetry)
        2. Straight flights — corrected (forced symmetry)
        3. Obstacle flights — uncorrected (loops reflect banking)
        4. Obstacle flights — corrected (forced symmetry)

    Args:
        scores_df: DataFrame of rotation-corrected scores.
        scores_df_norot: DataFrame of uncorrected (pre-rotation) scores.
        alpha_straight: Scatter-point opacity for straight flights. Defaults to 0.5.
        alpha_obstacle: Scatter-point opacity for obstacle flights. Defaults to 0.8.

    Returns:
        Tuple of (fig, axs) where axs has shape (4, 2).
    """
    # Uncorrected straight flights
    lr_straight_uncorr, score_5, score_95 = prepare_left_right_comparison(
        scores_df_norot, obstacle=0, perchDist=["9", "12"], horzdist="first_half"
    )
    # Corrected straight flights
    lr_straight_corr, _, _ = prepare_left_right_comparison(
        scores_df, obstacle=0, perchDist=["9", "12"], horzdist="first_half"
    )
    # Uncorrected obstacle flights
    lr_obstacle_uncorr, _, _ = prepare_left_right_comparison(
        scores_df_norot, obstacle=1
    )
    # Corrected obstacle flights
    lr_obstacle_corr, _, _ = prepare_left_right_comparison(
        scores_df, obstacle=1
    )

    conditions = [
        {"title": "Straight flights — uncorrected (underlying symmetry)", "left_right_scores": lr_straight_uncorr},
        {"title": "Straight flights — corrected (forced symmetry)", "left_right_scores": lr_straight_corr},
        {"title": "Obstacle flights — uncorrected (loops reflect banking)", "left_right_scores": lr_obstacle_uncorr},
        {"title": "Obstacle flights — corrected (forced symmetry)", "left_right_scores": lr_obstacle_corr},
    ]

    fig, axs = plot_left_right_just_two_stacked(
        conditions, score_5, score_95,
        alpha=[alpha_straight, alpha_straight, alpha_obstacle, alpha_obstacle]
    )

    return fig, axs


def plot_left_right_empty(
    score_5, score_95, PC=0, bkgrd_color='white', figsize=(2, 2)
):
    """Create empty symmetry panel with reference lines for legend/schematic.

    Draws the line of perfect symmetry (y = x, solid grey) and an offset
    guide line (dotted) without any data. Useful as an explanatory panel
    showing what the symmetry scatter plots represent.

    Args:
        score_5: Per-PC lower axis limits (dict or Series indexed by PC name).
        score_95: Per-PC upper axis limits (dict or Series indexed by PC name).
        PC: Zero-indexed PC number for which to draw the panel. Defaults to 0
            (PC1).
        bkgrd_color: Background colour of the panel. Defaults to 'white'.
        figsize: Figure size (width, height) in inches. Defaults to (2, 2).

    Returns:
        Tuple of (fig, ax).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get min and max values for this PC
    min_val = score_5[f'PC{PC+1:02}']
    max_val = score_95[f'PC{PC+1:02}']

    # Create slightly offset dotted line (20% above diagonal)
    offset = (max_val - min_val)+0.3
    x_grid = np.linspace(min_val, max_val, 100)
    dotted_line = x_grid + offset

    # Plot the diagonal and dotted lines
    ax.plot([min_val, max_val], [min_val, max_val], '-', c='grey', linewidth=0.8)
    ax.plot(x_grid, dotted_line, ':', c='black', linewidth=0.8)

    # Add "line of symmetry" text along diagonal
    # Calculate middle point of the line
    mid_x = (min_val + max_val) / 2
    mid_y = mid_x  # Since it's on the diagonal

    # Add text with rotation to match diagonal
    ax.text(mid_x, mid_y, 'line of symmetry',
           rotation=45,  # Rotate 45 degrees to match diagonal
           ha='center',  # Horizontal alignment
           va='center',  # Vertical alignment
           transform=ax.transData)  # Use data coordinates



    # Set background color
    ax.set_facecolor(bkgrd_color)

    # Axis limits and ticks
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks([min_val, 0, max_val])
    ax.set_yticks([0, max_val])
    ax.set_xticklabels("")
    ax.set_yticklabels("")

    # Labels
    ax.set_ylabel('left scores', fontsize=8)
    ax.set_xlabel('right scores', fontsize=8)
    # Grid
    ax.grid(True)

    fig.tight_layout()

    return fig, ax


def plot_symmetry_scores(symmetry_scores, threshold=0.05):
    """Plot per-component asymmetry scores with a significance threshold line.

    Draws a scatter of asymmetry scores (one point per PC) and a horizontal
    dashed red line at the threshold value. Components above the threshold are
    considered meaningfully asymmetric.

    Args:
        symmetry_scores: Asymmetry score for each principal component, length 12.
        threshold: Threshold value drawn as a dashed red line. Defaults to 0.05.

    Returns:
        Tuple of (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(
        np.arange(1, len(symmetry_scores) + 1),
        symmetry_scores,
        color='black',
        s=5,
    )
    ax.axhline(threshold, color='red', linestyle='--')
    ax.set_xticks(np.arange(1, len(symmetry_scores)+1))
    ax.set_xticklabels(np.arange(1, len(symmetry_scores)+1), fontsize=8, rotation=45)
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Asymmetry score')

    return fig, ax
