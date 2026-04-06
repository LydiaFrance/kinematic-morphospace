"""Left-right symmetry visualisation for bilateral PC scores."""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from ..pca_scores import get_binned_scores
from ..data_filtering import filter_by
from .markers import plot_raw_markers


def prepare_left_right_comparison(scores_df, **filters):
    """Prepare merged left/right PC scores for bilateral symmetry analysis.

    Filters left- and right-wing scores separately, merges them on frameID
    into a single row per frame, and computes axis-limit percentiles for
    consistent cross-panel scaling. Manual overrides for specific PCs ensure
    that all panels share a visually consistent scale.

    Args:
        scores_df: DataFrame containing PC scores and metadata; must include
            frameID, seqID, and left columns.
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
    left_right_scores = left_scores.merge(right_scores, left_index=True, right_index=True, suffixes=('_left', '_right'))

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
                 colour, alpha, bkgrd_color='white'):
    """Render a single left-vs-right scatter panel for one PC."""
    pc_label = f'PC{PC+1:02}'

    data = np.array([left_right_scores[f'{pc_label}_right'],
                     left_right_scores[f'{pc_label}_left']]).T
    major_slope, major_intercept, percentage_variance = _major_axis_regression(data)
    x_grid, major_line = _major_axis_line(data, major_slope)

    plot_raw_markers(ax,
                     left_right_scores[f'{pc_label}_right'],
                     left_right_scores[f'{pc_label}_left'],
                     colour=colour, alpha=alpha)

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


def plot_left_right(left_right_scores, score_5, score_95, alpha=0.05, bkgrd_color='white'):
    """Plot a 4x3 grid comparing left vs right PC scores for all 12 morphing modes.

    Each of the 12 panels shows a scatter of left-wing scores against right-wing
    scores for one PC, with the major-axis regression line and the line of perfect
    symmetry (y = x) overlaid. Deviation from the diagonal reveals left-right
    asymmetry in that morphing mode.

    Args:
        left_right_scores: Merged left/right DataFrame as returned by
            prepare_left_right_comparison().
        score_5: Per-PC lower axis limits.
        score_95: Per-PC upper axis limits.
        alpha: Scatter-point opacity. Defaults to 0.05.
        bkgrd_color: Background colour for each panel. Defaults to 'white'.

    Returns:
        Tuple of (fig, axs) where axs is a flat array of 12 Axes.
    """
    fig, axs = plt.subplots(4, 3, figsize=(8, 8),
                            sharex=False, sharey=False,
                            gridspec_kw={'hspace': 0.15, 'wspace': 0})
    axs = axs.flatten()
    colour_list = ['#B5E675', '#6ED8A9', '#51B3D4',
              '#4579AA', '#F19EBA', '#BC96C9',
              '#917AC2', '#BE607F', '#624E8B',
              '#888888', '#888888', '#888888']

    for PC in range(12):
        _plot_one_pc(axs[PC], left_right_scores, score_5, score_95,
                     PC, colour_list[PC], alpha, bkgrd_color)

    return fig, axs


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
        Tuple of (fig, axs) where axs is a flat array of 12 Axes (10 hidden).
    """
    fig, axs = plt.subplots(4, 3, figsize=(8, 8),
                            sharex=False, sharey=False,
                            gridspec_kw={'hspace': 0.15, 'wspace': 0})
    axs = axs.flatten()
    colour_list = ['#B5E675', '#6ED8A9']

    for PC in range(2):
        _plot_one_pc(axs[PC], left_right_scores, score_5, score_95,
                     PC, colour_list[PC], alpha)

    for PC in range(2, 12):
        axs[PC].axis('off')

    return fig, axs


def plot_left_right_empty(score_5, score_95, PC=0, bkgrd_color='white', figsize=(2, 2)):
    """Create an empty symmetry panel with reference lines for use as a legend or schematic.

    Draws the line of perfect symmetry (y = x, solid grey) and an offset guide
    line (dotted) without any data. Useful as an explanatory panel showing
    what the symmetry scatter plots represent.

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
    ax.set_ylabel(f'left scores', fontsize=8)
    ax.set_xlabel(f'right scores', fontsize=8)
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
    ax.scatter(np.arange(1, len(symmetry_scores)+1), symmetry_scores, color='black', s=5)
    ax.axhline(threshold, color='red', linestyle='--')
    ax.set_xticks(np.arange(1, len(symmetry_scores)+1))
    ax.set_xticklabels(np.arange(1, len(symmetry_scores)+1), fontsize=8, rotation=45)
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Asymmetry score')

    return fig, ax
