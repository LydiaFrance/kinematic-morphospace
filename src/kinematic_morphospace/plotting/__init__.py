"""kinematic_morphospace.plotting — visualisation subpackage for kinematic-morphospace figures."""

import os
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def save_figure(fig, filepath, dpi=300, rasterize=False):
    """Save a matplotlib figure to PDF or PNG, accepting several figure types.

    Accepts a bare Figure, a (Figure, Axes) tuple returned by most plotting
    helpers, or a bare Axes (get_figure() is called automatically). Parent
    directories are created on demand.

    Args:
        fig: The figure to save. May be a Figure, a (Figure, Axes) tuple,
            or a bare Axes object.
        filepath: Destination path; the file extension determines the format.
        dpi: Resolution for raster output. Defaults to 300.
        rasterize: When True, all axes content is rasterised before saving
            as PDF. This embeds a high-resolution bitmap inside the PDF,
            keeping file sizes small for scatter-heavy figures while
            preserving vector text and axes. Defaults to False.
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
    plot_symmetry_scores, summarise_symmetry, print_symmetry_summary)

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

# Robustness-validation schematics
from .schematics import (
    plot_shuffle_schematic, plot_subsampling_schematic,
    plot_pairwise_distance_schematic, plot_relabelling_schematic,
    plot_imputation_schematic, MARKER_COLOURS,
    _layout_shuffle_schematic, _layout_subsampling_schematic,
    _layout_relabelling_schematic, _layout_imputation_schematic,
    _layout_pairwise_distance_schematic,
    plot_autocorrelation_schematic)

# Individual-vs-shared subspace comparisons
from .subspaces import (
    plot_cosine_profile, plot_bootstrap_cosines, plot_method_comparison,
    HAWK_COLOURS)

# Flight-behaviour continuum (NB12)
from .continuum import (
    compute_flight_phase_traces,
    plot_flight_phase_overlay, plot_transition_overlay,
    plot_flight_phase_time_traces,
    plot_bic_sweep, plot_continuum_summary)

# Robustness composite figures (schematic + results)
from .robustness import (
    plot_shuffle_composite, plot_subsampling_composite,
    plot_pairwise_composite, plot_relabelling_composite,
    plot_imputation_composite, plot_hull_coverage, plot_hull_outlier_markers,
    plot_occlusion_bias)
