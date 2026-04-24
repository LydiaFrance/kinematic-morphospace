"""kinematic_morphospace.plotting — visualisation subpackage."""

import os
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Trajectories and whole-body kinematics
from .angles import bin_and_plot as bin_and_plot
from .angles import plot_angles_by_distance as plot_angles_by_distance
from .angles import plot_whole_body_angles as plot_whole_body_angles

# Clustering visualisation
from .clusters import get_cluster_colours as get_cluster_colours
from .clusters import get_cluster_counts as get_cluster_counts
from .clusters import plot_cluster_counts as plot_cluster_counts
from .clusters import plot_cluster_diffs as plot_cluster_diffs
from .clusters import plot_cluster_experience_diffs as plot_cluster_experience_diffs
from .clusters import plot_cluster_size_distribution as plot_cluster_size_distribution
from .clusters import plot_clusters as plot_clusters
from .clusters import (
    plot_marker_errors_with_thresholds as plot_marker_errors_with_thresholds,
)
from .clusters import plot_reconstruction_errors as plot_reconstruction_errors

# PCA component loadings
from .components import compare_coeffs_grid as compare_coeffs_grid
from .components import compare_coeffs_hawks as compare_coeffs_hawks
from .components import plot_compare_components_grid as plot_compare_components_grid
from .components import plot_components_grid as plot_components_grid

# Condition comparisons (obstacle, weight, experience)
from .conditions import plot_score_naive_control as plot_score_naive_control
from .conditions import plot_score_obstacle_control as plot_score_obstacle_control
from .conditions import plot_score_weight_control as plot_score_weight_control

# Flight-behaviour continuum (NB12)
from .continuum import compute_flight_phase_traces as compute_flight_phase_traces
from .continuum import plot_bic_sweep as plot_bic_sweep
from .continuum import plot_continuum_summary as plot_continuum_summary
from .continuum import plot_flight_phase_overlay as plot_flight_phase_overlay
from .continuum import plot_flight_phase_time_traces as plot_flight_phase_time_traces
from .continuum import plot_transition_overlay as plot_transition_overlay

# Score heatmaps
from .heatmaps import (
    plot_difference_exp_scores_heatmap as plot_difference_exp_scores_heatmap,
)
from .heatmaps import (
    plot_difference_PC_scores_heatmap as plot_difference_PC_scores_heatmap,
)
from .heatmaps import plot_PC_score_heatmaps as plot_PC_score_heatmaps
from .heatmaps import prepare_heatmap_comparison as prepare_heatmap_comparison

# Raw marker scatter plots
from .markers import plot_3d_scatter as plot_3d_scatter
from .markers import plot_3d_scatter_with_animation as plot_3d_scatter_with_animation
from .markers import plot_bird_marker_comparisons as plot_bird_marker_comparisons
from .markers import plot_raw_markers as plot_raw_markers
from .markers import plot_uncorrected_markers as plot_uncorrected_markers

# Robustness composite figures (schematic + results)
from .robustness import plot_hull_coverage as plot_hull_coverage
from .robustness import plot_hull_outlier_markers as plot_hull_outlier_markers
from .robustness import plot_imputation_composite as plot_imputation_composite
from .robustness import plot_occlusion_bias as plot_occlusion_bias
from .robustness import plot_pairwise_composite as plot_pairwise_composite
from .robustness import plot_relabelling_composite as plot_relabelling_composite
from .robustness import plot_shuffle_composite as plot_shuffle_composite
from .robustness import plot_subsampling_composite as plot_subsampling_composite

# Robustness-validation schematics
from .schematics import MARKER_COLOURS as MARKER_COLOURS
from .schematics import _layout_imputation_schematic as _layout_imputation_schematic
from .schematics import (
    _layout_pairwise_distance_schematic as _layout_pairwise_distance_schematic,
)
from .schematics import _layout_relabelling_schematic as _layout_relabelling_schematic
from .schematics import _layout_shuffle_schematic as _layout_shuffle_schematic
from .schematics import _layout_subsampling_schematic as _layout_subsampling_schematic
from .schematics import plot_autocorrelation_schematic as plot_autocorrelation_schematic
from .schematics import plot_imputation_schematic as plot_imputation_schematic
from .schematics import (
    plot_pairwise_distance_schematic as plot_pairwise_distance_schematic,
)
from .schematics import plot_relabelling_schematic as plot_relabelling_schematic
from .schematics import plot_shuffle_schematic as plot_shuffle_schematic
from .schematics import plot_subsampling_schematic as plot_subsampling_schematic

# PC score time traces
from .scores import plot_pc_comparison_grid as plot_pc_comparison_grid
from .scores import plot_score as plot_score
from .scores import plot_left_right_timeseries as plot_left_right_timeseries
from .scores import plot_score_multi_bird as plot_score_multi_bird
from .scores import plot_score_multi_distance as plot_score_multi_distance
from .scores import plot_score_multi_PCs as plot_score_multi_PCs

# Cross-species marker visualisation
from .species import plot_bird_markers as plot_bird_markers
from .species import prepare_long_neck_bird as prepare_long_neck_bird

# Individual-vs-shared subspace comparisons
from .subspaces import HAWK_COLOURS as HAWK_COLOURS
from .subspaces import plot_bootstrap_cosines as plot_bootstrap_cosines
from .subspaces import plot_cosine_profile as plot_cosine_profile
from .subspaces import plot_method_comparison as plot_method_comparison

# Missingness analysis
from .missingness import plot_cooccurrence_heatmap as plot_cooccurrence_heatmap
from .missingness import plot_coverage_comparison as plot_coverage_comparison
from .missingness import plot_density_comparison as plot_density_comparison
from .missingness import plot_dropout_positions as plot_dropout_positions
from .missingness import plot_ordering_violations as plot_ordering_violations

# Left-right symmetry
from .symmetry import plot_left_right as plot_left_right
from .symmetry import plot_left_right_empty as plot_left_right_empty
from .symmetry import plot_left_right_just_two as plot_left_right_just_two
from .symmetry import plot_left_right_just_two_stacked as plot_left_right_just_two_stacked
from .symmetry import (
    plot_rotation_correction_comparison as plot_rotation_correction_comparison,
)
from .symmetry import plot_symmetry_validation as plot_symmetry_validation
from .symmetry import plot_symmetry_scores as plot_symmetry_scores
from .symmetry import prepare_left_right_comparison as prepare_left_right_comparison
from .symmetry import print_symmetry_summary as print_symmetry_summary
from .symmetry import summarise_symmetry as summarise_symmetry

# Trajectory plots
from .trajectories import plot_traj as plot_traj
from .trajectories import plot_traj_pair as plot_traj_pair
from .trajectories import plot_traj_scatter as plot_traj_scatter
from .trajectories import plot_trajectory_data as plot_trajectory_data
from .trajectories import save_hybrid_figure as save_hybrid_figure
from .trajectories import setup_trajectory_axis as setup_trajectory_axis

# Explained variance and scree plots
from .variance import (
    calculate_cumulative_variance_ratios as calculate_cumulative_variance_ratios,
)
from .variance import plot_cumulative_variance_ratios as plot_cumulative_variance_ratios
from .variance import plot_explained as plot_explained
from .variance import plot_explained_comparison as plot_explained_comparison
from .variance import plot_explained_overlay as plot_explained_overlay
from .variance import plot_hist_similar_shapes as plot_hist_similar_shapes
from .variance import plot_reconstruction_accuracy as plot_reconstruction_accuracy
from .variance import (
    table_cumulative_variance_ratios as table_cumulative_variance_ratios,
)


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
        msg = f"Expected Figure, got {type(fig)}"
        raise TypeError(msg)

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
