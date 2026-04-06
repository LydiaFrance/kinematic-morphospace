"""kinematic-morphospace: PCA-based morphospace analysis of bird wings and tails in flight."""
from __future__ import annotations

# --- Data layer ---
from .data_loading import (
    load_data as load_data,
    remove_frames as remove_frames,
    prepare_marker_data as prepare_marker_data,
    process_data as process_data,
    bin_markers_by_distance as bin_markers_by_distance,
)

from .data_scaling import (
    scale_data, unscale_data, add_turn_info, add_tailpack_data)

from .data_filtering import filter_by

# --- PCA core ---
from .pca_core import run_PCA, run_PCA_birds

from .pca_scores import (
    SIGN_CONVENTIONS, apply_sign_conventions,
    get_score_range, bin_by_horz_distance,
    get_binned_scores, get_score_df, prepare_heatmap_mode_order)

from .pca_reconstruct import reconstruct, to_bilateral, to_unilateral

# --- Rotation and symmetry ---
from .rotation import (
    assess_symmetry, vectorised_kabsch,
    extract_euler_angles_from_matrices, apply_rotation,
    undo_body_pitch_rotation, undo_body_rotation)

# --- Validation and statistics ---
from .validation import (
    bootstrapping_pca, pca_suitability_test, kmo_test, print_phase_pca_summary,
    reconstruction_rmse, compute_cosine_sweep,
)
from .null_testing import principal_cosines, pairwise_distance_features
from .display import (
    print_cev_comparison as print_cev_comparison,
    print_occlusion_pca as print_occlusion_pca,
    print_projection_validation as print_projection_validation,
    print_thinned_pca as print_thinned_pca,
    print_bootstrap_stability as print_bootstrap_stability,
    print_residual_eigenvalues as print_residual_eigenvalues,
    print_quintile_rmse as print_quintile_rmse,
    print_local_pca_stability as print_local_pca_stability,
    print_intrinsic_dimensionality as print_intrinsic_dimensionality,
    print_flight_phase_trace_summary as print_flight_phase_trace_summary,
    print_overlap_metrics as print_overlap_metrics,
    print_bic_summary as print_bic_summary,
    print_method_comparison as print_method_comparison,
    print_complete_markers_summary as print_complete_markers_summary,
    print_partial_markers_summary as print_partial_markers_summary,
    print_dataset_split as print_dataset_split,
    print_density_shift_table as print_density_shift_table,
    print_marker_dropout_rates as print_marker_dropout_rates,
    print_anatomical_violations as print_anatomical_violations,
    print_violation_breakdown as print_violation_breakdown,
)

# --- Labelling and clustering ---
from .labelling import (
    lower_dim_reconstruction, calculate_reconstruction_errors,
    calculate_marker_thresholds, filter_low_error_frames,
    clustering_analysis, kmeans_clustering, analyse_clusters)

from .clustering import (
    get_cluster_labels, restrict_cluster_labels,
    reorder_cluster_labels, get_cluster_counts)

# --- Cross-species ---
from .cross_species import (
    load_harvey_data, select_max_wingspan_row, clean_body_data,
    process_body_bird_id, merge_bird_data, filter_marker_columns,
    set_new_origin_and_axes, compute_derived_markers,
    check_and_fix_shoulder_distance, fix_leftright_sign,
    integrate_dataframe_to_bird3D)

from .species_transform import (
    create_marker_dict, transform_hawk_to_species,
    transform_principal_components)

__all__ = [
    # Data layer
    "load_data", "remove_frames", "prepare_marker_data", "process_data",
    "bin_markers_by_distance",
    "scale_data", "unscale_data", "add_turn_info", "add_tailpack_data",
    "filter_by",
    # PCA core
    "run_PCA", "run_PCA_birds",
    "SIGN_CONVENTIONS", "apply_sign_conventions",
    "get_score_range", "bin_by_horz_distance",
    "get_binned_scores", "get_score_df", "prepare_heatmap_mode_order",
    "reconstruct", "to_bilateral", "to_unilateral",
    # Rotation and symmetry
    "assess_symmetry", "vectorised_kabsch",
    "extract_euler_angles_from_matrices", "apply_rotation",
    "undo_body_pitch_rotation", "undo_body_rotation",
    # Validation and statistics
    "bootstrapping_pca", "pca_suitability_test", "kmo_test", "print_phase_pca_summary",
    "reconstruction_rmse", "compute_cosine_sweep",
    "principal_cosines", "pairwise_distance_features",
    "print_cev_comparison", "print_occlusion_pca", "print_projection_validation",
    "print_thinned_pca", "print_bootstrap_stability",
    "print_residual_eigenvalues", "print_quintile_rmse",
    "print_local_pca_stability", "print_intrinsic_dimensionality",
    "print_method_comparison",
    "print_complete_markers_summary", "print_partial_markers_summary",
    "print_dataset_split", "print_density_shift_table",
    "print_marker_dropout_rates", "print_anatomical_violations", "print_violation_breakdown",
    # Labelling and clustering
    "lower_dim_reconstruction", "calculate_reconstruction_errors",
    "calculate_marker_thresholds", "filter_low_error_frames",
    "clustering_analysis", "kmeans_clustering", "analyse_clusters",
    "get_cluster_labels", "restrict_cluster_labels",
    "reorder_cluster_labels", "get_cluster_counts",
    # Cross-species
    "load_harvey_data", "select_max_wingspan_row", "clean_body_data",
    "process_body_bird_id", "merge_bird_data", "filter_marker_columns",
    "set_new_origin_and_axes", "compute_derived_markers",
    "check_and_fix_shoulder_distance", "fix_leftright_sign",
    "integrate_dataframe_to_bird3D",
    "create_marker_dict", "transform_hawk_to_species",
    "transform_principal_components",
]

# --- Visualisation (from plotting subpackage, requires [plot] extra) ---
try:
    from .plotting import (
        plot_raw_markers, plot_components_grid, plot_explained,
        plot_score, plot_score_multi_distance, plot_score_multi_PCs,
        compare_coeffs_grid, compare_coeffs_hawks,
        plot_PC_score_heatmaps, plot_difference_PC_scores_heatmap,
        plot_left_right, plot_traj, plot_bird_markers,
        plot_reconstruction_errors, plot_marker_errors_with_thresholds,
        plot_cluster_size_distribution,
        plot_method_comparison)

    __all__ += [
        "plot_raw_markers", "plot_components_grid", "plot_explained",
        "plot_score", "plot_score_multi_distance", "plot_score_multi_PCs",
        "compare_coeffs_grid", "compare_coeffs_hawks",
        "plot_PC_score_heatmaps", "plot_difference_PC_scores_heatmap",
        "plot_left_right", "plot_traj", "plot_bird_markers",
        "plot_reconstruction_errors", "plot_marker_errors_with_thresholds",
        "plot_cluster_size_distribution",
        "plot_method_comparison",
    ]
except ImportError:
    pass

# --- Preprocessing (requires [preprocessing] extra) ---
try:
    from . import preprocessing  # noqa: F401

    __all__ += ["preprocessing"]
except ImportError:
    pass

try:
    from importlib.metadata import version
    __version__ = version(__name__)
except Exception:
    __version__ = "0.1.0"
