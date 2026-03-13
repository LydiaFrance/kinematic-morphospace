"""
kinematic-morphospace preprocessing: convert raw MATLAB data to analysis-ready CSVs.

This subpackage reproduces two MATLAB scripts as clean, testable Python
pipelines:

1. ``fix_data_2024_03_23.m`` — downstream processing (MAT/CSV → analysis
   tables). Entry point: :func:`run_from_csvs` or :func:`run_preprocessing`.

2. ``run_mocap_processing.m`` — upstream C3D processing (raw C3D → processed
   marker tables). Entry point: :func:`run_from_c3d`.

Quick start (recommended — from MATLAB-exported CSVs)::

    from kinematic_morphospace.preprocessing import run_from_csvs

    tables = run_from_csvs("/path/to/csv/dir")

Alternative (from .mat files — requires struct-based variables, not tables)::

    from kinematic_morphospace.preprocessing import PreprocessingConfig, run_preprocessing

    config = PreprocessingConfig(
        data_dir_2020="/path/to/2020/mat/files",
        data_dir_2017_body="/path/to/2017/mat/files",
        output_dir="/path/to/output",
    )
    tables = run_preprocessing(config)

From raw C3D files (requires ``ezc3d``)::

    from kinematic_morphospace.preprocessing import C3DConfig, run_from_c3d

    config = C3DConfig(mocap_folder="/path/to/c3d/files")
    tables = run_from_c3d(config)

Requires the ``[preprocessing]`` extra::

    pip install kinematic-morphospace[preprocessing]
"""
from __future__ import annotations

from .body_frame import estimate_body_pitch
from .body_rotation import (
    apply_rotation,
    build_body_frame,
    build_rotation_matrices,
    compute_pitch_angle,
    compute_yaw_angle,
    extract_body_angles,
    rotate_to_body_frame,
)
from .c3d_loader import BIRD_ID_MAP, build_file_list, filter_file_list, load_c3d
from .c3d_pipeline import C3DConfig, run_from_c3d, run_single_c3d
from .calibration import (
    apply_time_offsets,
    calibrate_position,
    calibrate_time,
    find_jump_frame,
)
from .coord_transform import (
    LEFT_PERCH,
    RIGHT_PERCH,
    compute_horizontal_distance,
    compute_relative_positions,
    detect_flight_direction,
    shift_origin_all_columns,
    shift_origin_to_perch,
)
from .harmonise import (
    add_metadata,
    extract_bird_id,
    extract_seq_id,
    harmonise_labelled,
    harmonise_trajectory,
    join_body_pitch,
    join_smooth_xyz,
)
from .marker_labelling import (
    BACKPACK_BINS,
    HEADPACK_BINS,
    TAILPACK_BINS,
    compute_pairwise_distances,
    filter_by_distance,
    fix_mislabelled_tailpack,
    label_body_markers,
)
from .mat_loader import (
    load_2017_data,
    load_2020_data,
    load_intermediate_csvs,
    load_mat,
    matlab_table_to_dataframe,
)
from .pipeline import PreprocessingConfig, run_from_csvs, run_preprocessing, save_csvs
from .duplicate_resolution import (
    detect_duplicates,
    resolve_duplicates,
    split_labelled_table,
)
from .polygon_labelling import label_by_polygons, load_polygon_boundaries
from .shape_tables import (
    create_bilateral_table,
    create_unilateral_table,
    filter_pure_side_frames,
    mirror_left_markers,
    pivot_markers_wide,
)
from .smoothing import (
    compute_body_statistics,
    moving_mean_smooth,
    smooth_spline,
    smooth_trajectory_with_gaps,
)
from .stationary import (
    compute_marker_movement,
    detect_stationary_markers,
    label_fixed_objects,
)
from .time_sync import create_time_variable, find_takeoff_frame
from .whole_body_pipeline import WholeBodyConfig, run_whole_body_analysis
from .trial_splitting import (
    detect_velocity_peaks,
    load_annotations,
    save_annotations,
    split_by_trial,
)

__all__ = [
    # Pipeline (downstream — CSV/MAT)
    "PreprocessingConfig",
    "run_preprocessing",
    "run_from_csvs",
    "save_csvs",
    # Pipeline (whole-body analysis)
    "WholeBodyConfig",
    "run_whole_body_analysis",
    # Pipeline (upstream — C3D)
    "C3DConfig",
    "run_from_c3d",
    "run_single_c3d",
    # C3D loading
    "load_c3d",
    "build_file_list",
    "filter_file_list",
    "BIRD_ID_MAP",
    # Stationary detection
    "compute_marker_movement",
    "detect_stationary_markers",
    "label_fixed_objects",
    # Trial splitting
    "detect_velocity_peaks",
    "load_annotations",
    "save_annotations",
    "split_by_trial",
    # Marker labelling
    "HEADPACK_BINS",
    "BACKPACK_BINS",
    "TAILPACK_BINS",
    "compute_pairwise_distances",
    "label_body_markers",
    "fix_mislabelled_tailpack",
    "filter_by_distance",
    # Polygon labelling
    "load_polygon_boundaries",
    "label_by_polygons",
    # Duplicate resolution
    "detect_duplicates",
    "resolve_duplicates",
    "split_labelled_table",
    # Smoothing
    "moving_mean_smooth",
    "smooth_spline",
    "smooth_trajectory_with_gaps",
    "compute_body_statistics",
    # Coordinate transforms
    "LEFT_PERCH",
    "RIGHT_PERCH",
    "detect_flight_direction",
    "shift_origin_to_perch",
    "shift_origin_all_columns",
    "compute_horizontal_distance",
    "compute_relative_positions",
    # Time sync
    "find_takeoff_frame",
    "create_time_variable",
    # Body frame (PCA-based)
    "estimate_body_pitch",
    # Body rotation (vectorized)
    "compute_pitch_angle",
    "compute_yaw_angle",
    "build_rotation_matrices",
    "apply_rotation",
    "build_body_frame",
    "rotate_to_body_frame",
    "extract_body_angles",
    # MAT loading
    "load_mat",
    "matlab_table_to_dataframe",
    "load_2020_data",
    "load_2017_data",
    "load_intermediate_csvs",
    # Harmonisation
    "harmonise_trajectory",
    "harmonise_labelled",
    "add_metadata",
    "extract_bird_id",
    "extract_seq_id",
    "join_body_pitch",
    "join_smooth_xyz",
    # Calibration
    "calibrate_position",
    "calibrate_time",
    "find_jump_frame",
    "apply_time_offsets",
    # Shape tables
    "create_unilateral_table",
    "create_bilateral_table",
    "pivot_markers_wide",
    "mirror_left_markers",
    "filter_pure_side_frames",
]
