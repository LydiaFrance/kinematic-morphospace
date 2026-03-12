"""Left-right symmetry visualisation for bilateral PC scores."""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

from ..pca_scores import get_binned_scores
from ..data_filtering import filter_by
from .markers import plot_raw_markers


def prepare_left_right_comparison(scores_df, **filters):
    """Prepare merged left/right PC scores for symmetry analysis.

    Filters left- and right-wing scores, merges them on ``frameID``,
    and computes axis-limit percentiles for consistent plotting.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and metadata (must include
        ``frameID``, ``seqID``, and ``left`` columns).
    **filters
        Keyword arguments forwarded to
        :func:`~kinematic_morphospace.data_filtering.filter_by` (e.g. ``obstacle``,
        ``horzdist``, ``hawkname``).

    Returns
    -------
    left_right_scores : pandas.DataFrame
        Merged DataFrame with ``_left`` / ``_right`` suffixed PC columns.
    score_5 : pandas.Series
        Lower axis-limit percentiles per PC.
    score_95 : pandas.Series
        Upper axis-limit percentiles per PC.
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
    """Fit a PCA major-axis line to 2D *data* (N×2).

    This is major-axis (MA) regression — the first eigenvector of the
    bivariate cloud — not reduced major-axis (RMA) regression.
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

def plot_left_right(left_right_scores, score_5, score_95, alpha=0.05, bkgrd_color='white'):
    """Plot a 4x3 grid comparing left vs right PC scores for all 12 PCs.

    Each panel shows a scatter of left-wing against right-wing scores
    with a PCA major-axis regression line and the line of perfect
    symmetry overlaid.

    Parameters
    ----------
    left_right_scores : pandas.DataFrame
        Merged left/right scores as returned by
        :func:`prepare_left_right_comparison`.
    score_5 : pandas.Series
        Lower axis limits per PC.
    score_95 : pandas.Series
        Upper axis limits per PC.
    alpha : float, optional
        Scatter-point transparency.
    bkgrd_color : str, optional
        Background colour for the panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Flat array of subplot axes.
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
    """Plot left-vs-right symmetry for PCs 1--2 only.

    Intended for showing pre-rotation-correction scores where the first
    two components have not yet been forced symmetrical. Remaining
    panels are hidden.

    Parameters
    ----------
    left_right_scores : pandas.DataFrame
        Merged left/right scores as returned by
        :func:`prepare_left_right_comparison`.
    score_5 : pandas.Series
        Lower axis limits per PC.
    score_95 : pandas.Series
        Upper axis limits per PC.
    alpha : float, optional
        Scatter-point transparency.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Flat array of subplot axes.
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
    """Create an empty symmetry panel with reference lines for annotation.

    Draws the line of perfect symmetry (solid) and an offset guide line
    (dotted) without any data, useful as a legend or schematic panel.

    Parameters
    ----------
    score_5 : dict or pandas.Series
        Lower axis limits per PC.
    score_95 : dict or pandas.Series
        Upper axis limits per PC.
    PC : int, optional
        Zero-indexed PC number (default 0 for PC1).
    bkgrd_color : str, optional
        Background colour of the panel.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
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
    """Plot per-component asymmetry scores with a significance threshold.

    Draws a scatter of asymmetry scores (one per PC) and a horizontal
    line indicating the threshold above which a component is considered
    meaningfully asymmetric.

    Parameters
    ----------
    symmetry_scores : array-like
        Asymmetry score for each principal component.
    threshold : float, optional
        Threshold value drawn as a horizontal dashed line.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(np.arange(1, len(symmetry_scores)+1), symmetry_scores, color='black', s=5)
    ax.axhline(threshold, color='red', linestyle='--')
    ax.set_xticks(np.arange(1, len(symmetry_scores)+1))
    ax.set_xticklabels(np.arange(1, len(symmetry_scores)+1), fontsize=8, rotation=45)
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Asymmetry score')

    return fig, ax
