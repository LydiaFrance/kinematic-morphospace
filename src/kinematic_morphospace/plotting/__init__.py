"""kinematic_morphospace.plotting — visualisation subpackage for kinematic-morphospace figures."""

import os
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def save_figure(fig, filepath, dpi=300, rasterize=False):
    """Save a matplotlib figure to PDF/PNG.

    Accepts fig as a Figure, (fig, axes) tuple, or bare Axes
    (calls get_figure()). Creates parent directories automatically.
    Defaults to tight bounding box.

    Parameters
    ----------
    fig : Figure, (Figure, Axes) tuple, or Axes
        The figure to save.
    filepath : str or Path
        Destination path (extension determines format).
    dpi : int
        Resolution for raster output (default 300).
    rasterize : bool
        If True, rasterize all artists before saving as PDF. This
        embeds a high-resolution bitmap inside the PDF, which keeps
        file sizes small for scatter-heavy figures while preserving
        vector text/axes.
    """
    # Unwrap common return types from plotting functions
    if isinstance(fig, tuple):
        fig = fig[0]
    if isinstance(fig, Axes):
        fig = fig.get_figure()
    if not isinstance(fig, Figure):
        raise TypeError(f"Expected Figure, got {type(fig)}")

    filepath = Path(filepath)
    os.makedirs(filepath.parent, exist_ok=True)

    if rasterize:
        # Rasterize all axes content for compact PDFs with dense data
        for ax in fig.get_axes():
            ax.set_rasterized(True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        # Restore to avoid side-effects if figure is reused
        for ax in fig.get_axes():
            ax.set_rasterized(False)
    else:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

    print(f"Saved: {filepath}")


# Trajectories and whole-body kinematics
from .trajectories import (
    plot_trajectory_data, plot_traj, save_hybrid_figure,
    setup_trajectory_axis, plot_traj_scatter)

# Body angles (roll, yaw, pitch)
from .angles import (
    bin_and_plot, plot_whole_body_angles, plot_angles_by_distance)

# Raw marker scatter plots
from .markers import (
    plot_raw_markers, plot_uncorrected_markers, plot_bird_marker_comparisons,
    plot_3d_scatter, plot_3d_scatter_with_animation)

# Explained variance and scree plots
from .variance import (
    plot_explained, table_cumulative_variance_ratios,
    calculate_cumulative_variance_ratios, plot_cumulative_variance_ratios,
    plot_explained_comparison, plot_hist_similar_shapes)

# PCA component loadings
from .components import (
    plot_components_grid, compare_coeffs_hawks, compare_coeffs_grid,
    plot_compare_components_grid)

# PC score time traces
from .scores import (
    plot_score, plot_score_multi_PCs, plot_score_multi_distance,
    plot_pc_comparison_grid, plot_score_multi_bird)

# Left-right symmetry
from .symmetry import (
    prepare_left_right_comparison, plot_left_right,
    plot_left_right_just_two, plot_left_right_empty,
    plot_symmetry_scores)

# Score heatmaps
from .heatmaps import (
    prepare_heatmap_comparison, plot_difference_PC_scores_heatmap,
    plot_PC_score_heatmaps, plot_difference_exp_scores_heatmap)

# Condition comparisons (obstacle, weight, experience)
from .conditions import (
    plot_score_obstacle_control, plot_score_weight_control,
    plot_score_naive_control)

# Clustering visualisation
from .clusters import (
    get_cluster_colours, plot_clusters, get_cluster_counts,
    plot_cluster_counts, plot_cluster_diffs, plot_cluster_experience_diffs,
    plot_reconstruction_errors, plot_marker_errors_with_thresholds,
    plot_cluster_size_distribution)

# Cross-species marker visualisation
from .species import plot_bird_markers
