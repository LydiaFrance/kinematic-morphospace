"""PCA-based morphospace analysis of bird wings and tails in flight."""
from __future__ import annotations

from .clustering import (
    get_cluster_counts,
    get_cluster_labels,
    reorder_cluster_labels,
    restrict_cluster_labels,
)

# --- Cross-species ---
from .cross_species import (
    check_and_fix_shoulder_distance,
    clean_body_data,
    compute_derived_markers,
    fill_missing_body_measurements,
    filter_marker_columns,
    fix_leftright_sign,
    integrate_dataframe_to_bird3D,
    load_harvey_data,
    merge_bird_data,
    process_body_bird_id,
    select_best_specimen_per_species,
    select_max_wingspan_row,
    set_new_origin_and_axes,
)
from .data_filtering import filter_by
from .data_loading import (
    bin_markers_by_distance as bin_markers_by_distance,
)

# --- Data layer ---
from .data_loading import (
    load_data as load_data,
)
from .data_loading import (
    prepare_marker_data as prepare_marker_data,
)
from .data_loading import (
    process_data as process_data,
)
from .data_loading import (
    remove_frames as remove_frames,
)
from .data_scaling import add_tailpack_data, add_turn_info, scale_data, unscale_data
from .display import (
    print_anatomical_violations as print_anatomical_violations,
)
from .display import (
    print_bic_summary as print_bic_summary,
)
from .display import (
    print_bootstrap_stability as print_bootstrap_stability,
)
from .display import (
    print_cev_comparison as print_cev_comparison,
)
from .display import (
    print_complete_markers_summary as print_complete_markers_summary,
)
from .display import (
    print_dataset_split as print_dataset_split,
)
from .display import (
    print_density_shift_table as print_density_shift_table,
)
from .display import (
    print_flight_phase_trace_summary as print_flight_phase_trace_summary,
)
from .display import (
    print_intrinsic_dimensionality as print_intrinsic_dimensionality,
)
from .display import (
    print_local_pca_stability as print_local_pca_stability,
)
from .display import (
    print_marker_dropout_rates as print_marker_dropout_rates,
)
from .display import (
    print_per_hawk_dropout as print_per_hawk_dropout,
)
from .display import (
    print_method_comparison as print_method_comparison,
)
from .display import (
    print_occlusion_pca as print_occlusion_pca,
)
from .display import (
    print_overlap_metrics as print_overlap_metrics,
)
from .display import (
    print_partial_markers_summary as print_partial_markers_summary,
)
from .display import (
    print_projection_validation as print_projection_validation,
)
from .display import (
    print_quintile_rmse as print_quintile_rmse,
)
from .display import (
    print_residual_eigenvalues as print_residual_eigenvalues,
)
from .display import (
    print_thinned_pca as print_thinned_pca,
)
from .display import (
    print_violation_breakdown as print_violation_breakdown,
)
from .display import (
    print_imputation_results as print_imputation_results,
)

# --- Labelling and clustering ---
from .labelling import (
    analyse_clusters,
    calculate_marker_thresholds,
    calculate_reconstruction_errors,
    clustering_analysis,
    filter_low_error_frames,
    kmeans_clustering,
    lower_dim_reconstruction,
)
from .null_testing import pairwise_distance_features, principal_cosines

# --- PCA core ---
from .pca_core import run_PCA, run_PCA_birds
from .pca_reconstruct import reconstruct, to_bilateral, to_unilateral
from .pca_scores import (
    SIGN_CONVENTIONS,
    apply_sign_conventions,
    bin_by_horz_distance,
    get_binned_scores,
    get_score_df,
    get_score_range,
    prepare_heatmap_mode_order,
)

# --- Rotation and symmetry ---
from .rotation import (
    apply_rotation,
    assess_symmetry,
    extract_euler_angles_from_matrices,
    undo_body_pitch_rotation,
    undo_body_rotation,
    vectorised_kabsch,
)
from .species_transform import (
    create_marker_dict,
    prepare_long_neck_bird,
    transform_hawk_to_species,
    transform_principal_components,
)

# --- Validation and statistics ---
from .validation import (
    bootstrapping_pca,
    compute_cosine_sweep,
    kmo_test,
    pca_suitability_test,
    print_phase_pca_summary,
    reconstruction_rmse,
)

__all__ = [
    "SIGN_CONVENTIONS",
    "add_tailpack_data",
    "add_turn_info",
    "analyse_clusters",
    "apply_rotation",
    "apply_sign_conventions",
    # Rotation and symmetry
    "assess_symmetry",
    "bin_by_horz_distance",
    "bin_markers_by_distance",
    # Validation and statistics
    "bootstrapping_pca",
    "calculate_marker_thresholds",
    "calculate_reconstruction_errors",
    "check_and_fix_shoulder_distance",
    "clean_body_data",
    "clustering_analysis",
    "compute_cosine_sweep",
    "compute_derived_markers",
    "create_marker_dict",
    "extract_euler_angles_from_matrices",
    "fill_missing_body_measurements",
    "filter_by",
    "filter_low_error_frames",
    "filter_marker_columns",
    "fix_leftright_sign",
    "get_binned_scores",
    "get_cluster_counts",
    "get_cluster_labels",
    "get_score_df",
    "get_score_range",
    "integrate_dataframe_to_bird3D",
    "kmeans_clustering",
    "kmo_test",
    # Data layer
    "load_data",
    # Cross-species
    "load_harvey_data",
    # Labelling and clustering
    "lower_dim_reconstruction",
    "merge_bird_data",
    "pairwise_distance_features",
    "pca_suitability_test",
    "prepare_heatmap_mode_order",
    "prepare_long_neck_bird",
    "prepare_marker_data",
    "principal_cosines",
    "print_anatomical_violations",
    "print_bootstrap_stability",
    "print_cev_comparison",
    "print_complete_markers_summary",
    "print_dataset_split",
    "print_density_shift_table",
    "print_intrinsic_dimensionality",
    "print_local_pca_stability",
    "print_marker_dropout_rates",
    "print_method_comparison",
    "print_occlusion_pca",
    "print_partial_markers_summary",
    "print_per_hawk_dropout",
    "print_phase_pca_summary",
    "print_projection_validation",
    "print_quintile_rmse",
    "print_residual_eigenvalues",
    "print_thinned_pca",
    "print_violation_breakdown",
    "process_body_bird_id",
    "process_data",
    "reconstruct",
    "reconstruction_rmse",
    "remove_frames",
    "reorder_cluster_labels",
    "restrict_cluster_labels",
    # PCA core
    "run_PCA",
    "run_PCA_birds",
    "scale_data",
    "select_best_specimen_per_species",
    "select_max_wingspan_row",
    "set_new_origin_and_axes",
    "to_bilateral",
    "to_unilateral",
    "transform_hawk_to_species",
    "transform_principal_components",
    "undo_body_pitch_rotation",
    "undo_body_rotation",
    "unscale_data",
    "vectorised_kabsch",
]

# --- Visualisation (from plotting subpackage, requires [plot] extra) ---
try:
    from .plotting import (
        compare_coeffs_grid,
        compare_coeffs_hawks,
        plot_bird_markers,
        plot_cluster_size_distribution,
        plot_components_grid,
        plot_difference_PC_scores_heatmap,
        plot_explained,
        plot_left_right,
        plot_marker_errors_with_thresholds,
        plot_method_comparison,
        plot_PC_score_heatmaps,
        plot_raw_markers,
        plot_reconstruction_errors,
        plot_score,
        plot_score_multi_distance,
        plot_score_multi_PCs,
        plot_traj,
    )

    __all__ += [
        "compare_coeffs_grid",
        "compare_coeffs_hawks",
        "plot_PC_score_heatmaps",
        "plot_bird_markers",
        "plot_cluster_size_distribution",
        "plot_components_grid",
        "plot_difference_PC_scores_heatmap",
        "plot_explained",
        "plot_left_right",
        "plot_marker_errors_with_thresholds",
        "plot_method_comparison",
        "plot_raw_markers",
        "plot_reconstruction_errors",
        "plot_score",
        "plot_score_multi_PCs",
        "plot_score_multi_distance",
        "plot_traj",
    ]
except ImportError:
    pass

# --- Preprocessing (requires [preprocessing] extra) ---
try:
    from . import preprocessing

    __all__ += ["preprocessing"]
except ImportError:
    pass

try:
    from importlib.metadata import version
    __version__ = version(__name__)
except Exception:
    __version__ = "0.1.0"
