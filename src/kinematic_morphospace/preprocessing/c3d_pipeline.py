"""
Orchestration pipeline for C3D → processed marker tables.

Chains all upstream processing steps (C3D loading, stationary detection,
trial splitting, body marker labelling, smoothing, coordinate transforms,
time sync, body pitch) into a single ``run_from_c3d()`` call.

This reproduces the full ``run_mocap_processing.m`` pipeline in Python.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .body_frame import estimate_body_pitch
from .c3d_loader import build_file_list, filter_file_list, load_c3d
from .coord_transform import (
    LEFT_PERCH,
    RIGHT_PERCH,
    compute_horizontal_distance,
    detect_flight_direction,
    shift_origin_to_perch,
)
from .marker_labelling import (
    BACKPACK_BINS,
    HEADPACK_BINS,
    TAILPACK_BINS,
    label_body_markers,
)
from .smoothing import compute_body_statistics
from .stationary import (
    DEFAULT_OBJECT_RANGES,
    detect_stationary_markers,
    label_fixed_objects,
)
from .time_sync import create_time_variable, find_takeoff_frame
from .trial_splitting import detect_velocity_peaks, split_by_trial

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class C3DConfig:
    """Configuration for the C3D processing pipeline.

    All magic numbers from ``run_mocap_processing.m`` are exposed as named
    fields with their original default values.
    """

    # --- Data paths ---
    mocap_folder: str | Path = ""
    """Directory containing bird-specific subdirectories with C3D files."""

    output_dir: str | Path | None = None
    """If set, save output CSVs to this directory."""

    # --- Stationary detection ---
    stationary_threshold: float = 0.001
    """Movement threshold (metres) for stationary markers."""

    n_outlier_passes: int = 3
    """Outlier removal passes within stationary cluster."""

    # --- Trial splitting ---
    min_peak_distance: int = 250
    """Minimum frames between velocity peaks."""

    min_peak_width: int = 150
    """Minimum peak width in frames."""

    min_peak_height: float = 0.01
    """Minimum velocity peak height (m/frame)."""

    smooth_fraction: float = 0.05
    """Smoothing window as fraction of total frames (5%)."""

    # --- Marker labelling ---
    headpack_bins: list[tuple[float, float]] = field(
        default_factory=lambda: list(HEADPACK_BINS)
    )
    backpack_bins: list[tuple[float, float]] = field(
        default_factory=lambda: list(BACKPACK_BINS)
    )
    tailpack_bins: list[tuple[float, float]] = field(
        default_factory=lambda: list(TAILPACK_BINS)
    )

    # --- Fixed object Y-ranges ---
    object_y_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_OBJECT_RANGES)
    )

    # --- Smoothing ---
    smooth_window: int = 10
    """Moving-mean window for body position smoothing (frames)."""

    frame_rate: float = 200.0
    """Recording frame rate in Hz."""

    # --- Coordinate transforms ---
    left_perch: list[float] = field(
        default_factory=lambda: LEFT_PERCH.tolist()
    )
    right_perch: list[float] = field(
        default_factory=lambda: RIGHT_PERCH.tolist()
    )

    # --- Time sync ---
    takeoff_y_range: tuple[float, float] = (-8.935, -8.5)
    """Y-position range for takeoff detection (after origin shift)."""

    takeoff_min_speed: float = 2.0
    """Minimum speed (m/s) for takeoff detection."""

    # --- Perch distance (metadata) ---
    perch_distance: float = 9.0
    """Distance between perches in metres."""


# ---------------------------------------------------------------------------
# Single-recording processing
# ---------------------------------------------------------------------------


def run_single_c3d(
    path: str | Path,
    config: C3DConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Process a single C3D recording through the full pipeline.

    Parameters
    ----------
    path : str or Path
        Path to a ``.c3d`` file.
    config : C3DConfig, optional
        Pipeline configuration. Uses defaults if not provided.

    Returns
    -------
    dict[str, pd.DataFrame]
        Processed tables:

        - ``"markers"`` — full marker table with labels and trial column
        - ``"body_stats"`` — per-frame body statistics
        - ``"body_pitch"`` — per-frame pitch estimates
        - ``"metadata"`` — single-row metadata table
    """
    if config is None:
        config = C3DConfig()

    import numpy as np

    logger.info("Processing: %s", Path(path).name)

    # Step 1: Load C3D
    logger.info("  Step 1: Loading C3D")
    df, meta = load_c3d(path)
    frame_rate = meta["frame_rate"]

    # Step 2: Stationary detection
    logger.info("  Step 2: Detecting stationary markers")
    is_stationary = detect_stationary_markers(
        df, threshold=config.stationary_threshold,
        n_outlier_passes=config.n_outlier_passes,
    )
    df["label_stationary"] = df["marker_id"].map(is_stationary).fillna(False)

    # Step 2b: Label fixed objects
    object_labels = label_fixed_objects(
        df, is_stationary, y_ranges=config.object_y_ranges,
    )
    df["object_label"] = df["marker_id"].map(object_labels).fillna("unknown")

    # Step 3: Trial splitting
    logger.info("  Step 3: Splitting trials")
    peaks = detect_velocity_peaks(
        df,
        min_peak_distance=config.min_peak_distance,
        min_peak_width=config.min_peak_width,
        min_peak_height=config.min_peak_height,
        smooth_fraction=config.smooth_fraction,
    )
    if not peaks.empty:
        annotations = peaks[["start_frame", "end_frame"]].to_dict("records")
        df = split_by_trial(df, annotations)
    else:
        df["trial"] = 0

    # Step 4: Body marker labelling
    logger.info("  Step 4: Labelling body markers")
    body_labels = label_body_markers(
        df, is_stationary,
        headpack_bins=config.headpack_bins,
        backpack_bins=config.backpack_bins,
        tailpack_bins=config.tailpack_bins,
    )
    df["body_label"] = df["marker_id"].map(body_labels).fillna("unlabelled")

    # Step 5: Body statistics (smoothing)
    logger.info("  Step 5: Computing body statistics")
    body_stats = compute_body_statistics(
        df, body_labels,
        smooth_window=config.smooth_window,
        frame_rate=frame_rate,
    )

    # Step 6: Coordinate transform
    logger.info("  Step 6: Coordinate transform")
    lp = np.array(config.left_perch)
    rp = np.array(config.right_perch)

    direction = detect_flight_direction(body_stats)
    df = shift_origin_to_perch(df, direction, left_perch=lp, right_perch=rp)
    body_stats = shift_origin_to_perch(
        body_stats, direction, left_perch=lp, right_perch=rp, y_column="smooth_Y",
    )
    body_stats["HorzDistance"] = compute_horizontal_distance(body_stats)

    # Step 7: Time sync
    logger.info("  Step 7: Time synchronisation")
    frame_zero = find_takeoff_frame(
        body_stats,
        y_range=config.takeoff_y_range,
        min_speed=config.takeoff_min_speed,
    )
    if frame_zero is not None:
        df = create_time_variable(df, frame_zero, frame_rate)
        body_stats = create_time_variable(body_stats, frame_zero, frame_rate)
    else:
        logger.warning("  No takeoff detected — time not set")

    # Step 8: Body pitch
    logger.info("  Step 8: Estimating body pitch")
    pitch_df = estimate_body_pitch(df, body_labels)

    # Metadata table
    metadata = pd.DataFrame([{
        "source_file": str(path),
        "frame_rate": frame_rate,
        "n_frames": meta["n_frames"],
        "n_markers": meta["n_markers"],
        "n_trials": int(df["trial"].max()) if "trial" in df.columns else 0,
        "direction": direction,
        "frame_zero": frame_zero,
        "perch_distance": config.perch_distance,
    }])

    logger.info("  Done: %d rows, %d trials", len(df), metadata["n_trials"].iloc[0])

    return {
        "markers": df,
        "body_stats": body_stats,
        "body_pitch": pitch_df,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Multi-file orchestration
# ---------------------------------------------------------------------------


def run_from_c3d(
    config: C3DConfig,
) -> dict[str, pd.DataFrame]:
    """Process all C3D files in a directory through the full pipeline.

    Scans the ``config.mocap_folder`` for C3D files, filters for recordings
    with wing markers and backpack, processes each one, and concatenates
    the results.

    Parameters
    ----------
    config : C3DConfig
        Pipeline configuration.

    Returns
    -------
    dict[str, pd.DataFrame]
        Concatenated output tables:

        - ``"markers"`` — all marker data with labels, trials, time
        - ``"body_stats"`` — per-frame body statistics for all recordings
        - ``"body_pitch"`` — per-frame pitch for all recordings
        - ``"metadata"`` — one row per recording
    """
    logger.info("=" * 60)
    logger.info("kinematic-morphospace C3D Pipeline")
    logger.info("=" * 60)

    # Build and filter file list
    file_list = build_file_list(config.mocap_folder)
    if file_list.empty:
        msg = f"No C3D files found in {config.mocap_folder}"
        raise FileNotFoundError(msg)

    filtered = filter_file_list(file_list)
    if filtered.empty:
        msg = "No recordings with wing markers + backpack found"
        raise ValueError(msg)

    logger.info("Processing %d recordings", len(filtered))

    all_markers = []
    all_body_stats = []
    all_pitch = []
    all_metadata = []

    for _, row in filtered.iterrows():
        try:
            result = run_single_c3d(row["path"], config)
            # Tag with recording info
            for key in ["markers", "body_stats", "body_pitch"]:
                result[key]["source_file"] = row["filename"]
                result[key]["bird"] = row["bird"]

            all_markers.append(result["markers"])
            all_body_stats.append(result["body_stats"])
            all_pitch.append(result["body_pitch"])
            all_metadata.append(result["metadata"])
        except Exception:
            logger.exception("  Failed to process %s", row["filename"])

    if not all_markers:
        msg = "No recordings processed successfully"
        raise RuntimeError(msg)

    results = {
        "markers": pd.concat(all_markers, ignore_index=True),
        "body_stats": pd.concat(all_body_stats, ignore_index=True),
        "body_pitch": pd.concat(all_pitch, ignore_index=True),
        "metadata": pd.concat(all_metadata, ignore_index=True),
    }

    # Save if output_dir specified
    if config.output_dir:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            path = out / f"c3d_{name}.csv"
            df.to_csv(path, index=False)
            logger.info("  Saved %s → %s (%d rows)", name, path.name, len(df))

    logger.info("=" * 60)
    logger.info("C3D pipeline complete")
    for name, df in results.items():
        logger.info("  %s: %d rows x %d cols", name, len(df), len(df.columns))
    logger.info("=" * 60)

    return results
