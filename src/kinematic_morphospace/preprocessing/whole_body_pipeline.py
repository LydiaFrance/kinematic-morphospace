"""
Whole-body analysis pipeline: raw marker data to body-frame coordinates.

Orchestrates the 17-step processing pipeline from
``run_whole_body_analysis.m``, converting raw labelled/unlabelled marker
data into body-frame-rotated marker positions with pitch/yaw/roll angles.

Entry point: :func:`run_whole_body_analysis`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .body_rotation import (
    apply_rotation,
    build_body_frame,
    build_rotation_matrices,
    compute_pitch_angle,
    compute_yaw_angle,
    extract_body_angles,
    rotate_to_body_frame,
)
from .coord_transform import compute_relative_positions
from .marker_labelling import filter_by_distance, fix_mislabelled_tailpack
from .smoothing import smooth_trajectory_with_gaps

logger = logging.getLogger(__name__)


@dataclass
class WholeBodyConfig:
    """Configuration for the whole-body analysis pipeline.

    All thresholds and physical constants are configurable, with defaults
    matching ``run_whole_body_analysis.m``.
    """

    # Polygon boundary file
    polygon_path: str | Path | None = None

    # Frame rate
    frame_rate: float = 200.0

    # Smoothing
    smooth_rms: float = 0.001
    max_gap_frames: int = 30

    # Gap filtering thresholds
    min_time: float = 0.0
    min_horz_dist: float = 0.3

    # Distance filtering ranges (min, max) for each marker type
    backpack_dist_range: tuple[float, float] = (0.0, 0.05)
    tailpack_dist_range: tuple[float, float] = (0.1, 0.4)
    headpack_dist_range: tuple[float, float] = (0.05, 0.07)

    # Maximum distance for initial unlabelled marker filtering
    max_unlabelled_dist: float = 0.6

    # Bird name mapping for polygon labelling
    bird_id_map: dict[str, str] = field(default_factory=lambda: {
        "01": "Drogon",
        "03": "Ruby",
        "04": "Toothless",
        "05": "Charmander",
    })

    # Flight phase boundaries for 9m flights (horizontal distance ranges)
    flight_phases: dict[int, tuple[float, float]] = field(default_factory=lambda: {
        1: (7.0, 8.5),   # Early flapping
        2: (5.0, 7.0),   # Mid flapping
        3: (3.5, 5.0),   # Late flapping
        4: (0.3, 3.5),   # Gliding
    })


def smooth_backpack_per_sequence(
    labelled_df: pd.DataFrame,
    config: WholeBodyConfig,
    *,
    marker_label: str = "backpack",
) -> pd.DataFrame:
    """Smooth a body marker (backpack/tailpack/headpack) per sequence.

    Groups by sequence, computes mean position per frame, then applies
    gap-aware spline smoothing.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Labelled marker table with columns: ``seqID``, ``frame``, ``time``,
        ``frameID``, ``label``, ``X``, ``Y``, ``Z``.
    config : WholeBodyConfig
        Pipeline configuration.
    marker_label : str
        Which body marker to smooth (default ``"backpack"``).

    Returns
    -------
    pd.DataFrame
        Smooth table with columns: ``frame``, ``time``, ``seqID``,
        ``frameID``, ``smooth_X``, ``smooth_Y``, ``smooth_Z``,
        ``vel_X``, ``vel_Y``, ``vel_Z``, ``horzDist``.
    """
    markers = labelled_df[
        labelled_df["label"].str.contains(marker_label, na=False)
    ].copy()

    results = []
    for seq_id, seq_group in markers.groupby("seqID"):
        # Mean position per frame
        frame_mean = (
            seq_group.groupby("frame")[["X", "Y", "Z"]]
            .mean()
            .reset_index()
        )
        frame_mean = frame_mean.sort_values("frame")

        # Get time for each frame
        time_map = (
            seq_group.drop_duplicates("frame")
            .set_index("frame")["time"]
        )
        frame_mean["time"] = frame_mean["frame"].map(time_map)
        frame_mean = frame_mean.dropna(subset=["time"])

        if len(frame_mean) < 4:
            continue

        time_arr = frame_mean["time"].values
        frames_arr = frame_mean["frame"].values.astype(int)
        xyz_arr = frame_mean[["X", "Y", "Z"]].values

        horz_dist = np.sqrt(xyz_arr[:, 0] ** 2 + xyz_arr[:, 1] ** 2)

        result = smooth_trajectory_with_gaps(
            time_arr, frames_arr, xyz_arr,
            rms=config.smooth_rms,
            frame_rate=config.frame_rate,
            max_gap_frames=config.max_gap_frames,
            min_time=config.min_time,
            min_horz_dist=config.min_horz_dist,
            horz_dist=horz_dist,
        )

        n = len(result["frames"])
        seq_df = pd.DataFrame({
            "frame": result["frames"],
            "time": result["time"],
            "seqID": seq_id,
            "smooth_X": result["smooth"][:, 0],
            "smooth_Y": result["smooth"][:, 1],
            "smooth_Z": result["smooth"][:, 2],
            "vel_X": result["velocity"][:, 0],
            "vel_Y": result["velocity"][:, 1],
            "vel_Z": result["velocity"][:, 2],
        })
        seq_df["frameID"] = seq_df["seqID"] + "_" + seq_df["frame"].apply(
            lambda f: f"{int(f):06d}"
        )
        seq_df["horzDist"] = np.sqrt(
            seq_df["smooth_X"] ** 2 + seq_df["smooth_Y"] ** 2
        )

        results.append(seq_df)

    if not results:
        return pd.DataFrame()

    smooth_table = pd.concat(results, ignore_index=True)
    logger.info("  Smoothed %s: %d rows across %d sequences",
                marker_label, len(smooth_table), len(results))
    return smooth_table


def run_whole_body_analysis(
    labelled_df: pd.DataFrame,
    unlabelled_df: pd.DataFrame,
    config: WholeBodyConfig,
    *,
    info_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Execute the whole-body analysis pipeline.

    Reproduces the 17-step pipeline from ``run_whole_body_analysis.m``:

    1. Smooth backpack per sequence (spline with gap detection)
    2. Combine sequences (already done if input is combined)
    3. Fix mislabelled tailpack -> headpack
    4. Compute relative positions (marker - smooth backpack)
    5-6. Label unlabelled markers by polygon boundaries (if polygon_path set)
    7. Distance-based filtering
    8. Re-smooth backpack/tailpack/headpack with new markers
    9. Compute body pitch from tailpack vector
    10. Join body pitch to all tables
    11. Rotate markers by pitch
    12. Compute yaw from head-tail vector
    13. Build body frame via cross products
    14. Rotate markers into body-fixed coordinates
    15. Extract pitch/yaw/roll Euler angles
    16. Label metadata (if info_df provided)
    17. Return results

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Combined labelled marker table with columns: ``seqID``, ``frame``,
        ``time``, ``frameID``, ``label``, ``X``, ``Y``, ``Z``.
    unlabelled_df : pd.DataFrame
        Combined unlabelled marker table (same columns, ``label`` = ``""``).
    config : WholeBodyConfig
        Pipeline configuration.
    info_df : pd.DataFrame, optional
        Sequence info table with ``seqID``, ``Obstacle``, ``IMU`` columns.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``"smooth_backpack"``, ``"smooth_tailpack"``, ``"smooth_headpack"``,
        ``"labelled"``, ``"body_frame"`` (with rotated coords and angles).
    """
    logger.info("=" * 60)
    logger.info("Whole-Body Analysis Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Smooth backpack
    # ------------------------------------------------------------------
    logger.info("Step 1: Smoothing backpack")
    smooth_bp = smooth_backpack_per_sequence(labelled_df, config, marker_label="backpack")

    # ------------------------------------------------------------------
    # Step 3: Fix mislabelled tailpack
    # ------------------------------------------------------------------
    logger.info("Step 3: Fixing mislabelled tailpack markers")

    # First compute relative positions for labelled markers
    labelled_with_rel = _add_relative_positions(labelled_df, smooth_bp)
    labelled_with_rel = fix_mislabelled_tailpack(
        labelled_with_rel,
        relative_y_col="xyz_2",
        label_col="label",
    )

    # ------------------------------------------------------------------
    # Step 4: Compute relative positions for unlabelled markers
    # ------------------------------------------------------------------
    logger.info("Step 4: Computing relative positions")
    unlabelled_with_rel = _add_relative_positions(unlabelled_df, smooth_bp)

    # ------------------------------------------------------------------
    # Steps 5-6: Polygon labelling (optional)
    # ------------------------------------------------------------------
    if config.polygon_path is not None:
        logger.info("Steps 5-6: Polygon labelling")
        from .polygon_labelling import label_by_polygons, load_polygon_boundaries

        boundaries = load_polygon_boundaries(config.polygon_path)

        # Remove markers too far from backpack
        dist = np.linalg.norm(
            unlabelled_with_rel[["xyz_1", "xyz_2", "xyz_3"]].values, axis=1
        )
        unlabelled_with_rel = unlabelled_with_rel[
            dist <= config.max_unlabelled_dist
        ].copy()

        unlabelled_with_rel = label_by_polygons(
            unlabelled_with_rel,
            boundaries,
            bird_id_map=config.bird_id_map,
        )

        # Move newly labelled markers to the labelled table
        newly_labelled = unlabelled_with_rel[
            unlabelled_with_rel["label"] != ""
        ]
        if len(newly_labelled) > 0:
            labelled_with_rel = pd.concat(
                [labelled_with_rel, newly_labelled], ignore_index=True
            )
            unlabelled_with_rel = unlabelled_with_rel[
                unlabelled_with_rel["label"] == ""
            ].copy()
            logger.info("  Moved %d markers to labelled table", len(newly_labelled))

    # ------------------------------------------------------------------
    # Step 7: Distance-based filtering
    # ------------------------------------------------------------------
    logger.info("Step 7: Distance-based filtering")
    for label, (d_min, d_max) in [
        ("backpack", config.backpack_dist_range),
        ("tailpack", config.tailpack_dist_range),
        ("headpack", config.headpack_dist_range),
    ]:
        labelled_with_rel = filter_by_distance(
            labelled_with_rel, label, d_min, d_max,
        )

    # ------------------------------------------------------------------
    # Step 8: Re-smooth with new markers
    # ------------------------------------------------------------------
    logger.info("Step 8: Re-smoothing body markers")
    smooth_bp = smooth_backpack_per_sequence(
        labelled_with_rel, config, marker_label="backpack"
    )
    smooth_tp = smooth_backpack_per_sequence(
        labelled_with_rel, config, marker_label="tailpack"
    )
    smooth_hp = smooth_backpack_per_sequence(
        labelled_with_rel, config, marker_label="headpack"
    )

    # Recompute relative positions with updated smooth backpack
    smooth_tp = _add_relative_to_smooth(smooth_tp, smooth_bp)
    smooth_hp = _add_relative_to_smooth(smooth_hp, smooth_bp)

    # ------------------------------------------------------------------
    # Step 9: Compute body pitch from tailpack vector
    # ------------------------------------------------------------------
    logger.info("Step 9: Computing body pitch")
    tail_xyz = smooth_tp[["xyz_1", "xyz_2", "xyz_3"]].values
    smooth_tp["body_pitch"] = compute_pitch_angle(tail_xyz)

    # Angle of attack from velocity
    if all(c in smooth_bp.columns for c in ["vel_X", "vel_Y", "vel_Z"]):
        vel_xyz = smooth_bp[["vel_X", "vel_Y", "vel_Z"]].values
        aoa = compute_pitch_angle(vel_xyz)
        aoa = np.where(aoa < 0, aoa + 180, aoa - 180)
        smooth_bp["angle_of_attack"] = aoa

    # ------------------------------------------------------------------
    # Step 10: Join body pitch to other tables
    # ------------------------------------------------------------------
    logger.info("Step 10: Joining body pitch")
    pitch_lookup = smooth_tp[["frameID", "body_pitch"]].drop_duplicates("frameID")

    smooth_bp = smooth_bp.merge(pitch_lookup, on="frameID", how="inner")
    smooth_hp = smooth_hp.merge(pitch_lookup, on="frameID", how="inner")

    # ------------------------------------------------------------------
    # Step 11: Rotate markers by pitch
    # ------------------------------------------------------------------
    logger.info("Step 11: Rotating by pitch angle")
    for table in [smooth_tp, smooth_hp]:
        if "xyz_1" in table.columns and "body_pitch" in table.columns:
            xyz = table[["xyz_1", "xyz_2", "xyz_3"]].values
            R = build_rotation_matrices(table["body_pitch"].values, axis="x")
            rot = apply_rotation(xyz, R)
            table["rot_xyz_1"] = rot[:, 0]
            table["rot_xyz_2"] = rot[:, 1]
            table["rot_xyz_3"] = rot[:, 2]

    # ------------------------------------------------------------------
    # Step 12: Compute yaw from head-tail vector
    # ------------------------------------------------------------------
    logger.info("Step 12: Computing yaw")
    if "rot_xyz_1" in smooth_hp.columns and "rot_xyz_1" in smooth_tp.columns:
        hp_tp = smooth_hp.merge(
            smooth_tp[["frameID", "rot_xyz_1", "rot_xyz_2", "rot_xyz_3"]],
            on="frameID",
            how="inner",
            suffixes=("_head", "_tail"),
        )
        if len(hp_tp) > 0:
            head_rot = hp_tp[["rot_xyz_1_head", "rot_xyz_2_head", "rot_xyz_3_head"]].values
            tail_rot = hp_tp[["rot_xyz_1_tail", "rot_xyz_2_tail", "rot_xyz_3_tail"]].values

            # Head-to-tail vector in rotated frame
            ht_vector = head_rot - tail_rot
            yaw_angles = compute_yaw_angle(ht_vector)

            hp_tp["body_yaw"] = yaw_angles
            smooth_hp = smooth_hp.merge(
                hp_tp[["frameID", "body_yaw"]].drop_duplicates("frameID"),
                on="frameID",
                how="left",
            )

    # ------------------------------------------------------------------
    # Steps 13-15: Body frame rotation and angle extraction
    # ------------------------------------------------------------------
    logger.info("Steps 13-15: Body frame rotation")
    if "xyz_1" in smooth_tp.columns:
        tail_vecs = smooth_tp[["xyz_1", "xyz_2", "xyz_3"]].values
        body_axis, sideways, upwards = build_body_frame(tail_vecs)

        # Join unlabelled markers to get body-frame coordinates
        unlabelled_with_bp = _add_relative_positions(unlabelled_with_rel, smooth_bp)
        body_frame_df = unlabelled_with_bp.merge(
            smooth_tp[["frameID"]].drop_duplicates(),
            on="frameID",
            how="inner",
        )

        if len(body_frame_df) > 0:
            # Map body frame axes to each marker row
            tp_frame_map = smooth_tp.set_index("frameID")[[]].copy()
            tp_frame_map["body_ax_1"] = body_axis[:, 0]
            tp_frame_map["body_ax_2"] = body_axis[:, 1]
            tp_frame_map["body_ax_3"] = body_axis[:, 2]
            tp_frame_map["sw_1"] = sideways[:, 0]
            tp_frame_map["sw_2"] = sideways[:, 1]
            tp_frame_map["sw_3"] = sideways[:, 2]
            tp_frame_map["up_1"] = upwards[:, 0]
            tp_frame_map["up_2"] = upwards[:, 1]
            tp_frame_map["up_3"] = upwards[:, 2]

            body_frame_df = body_frame_df.merge(
                tp_frame_map.reset_index().drop_duplicates("frameID"),
                on="frameID",
                how="inner",
            )

            if len(body_frame_df) > 0 and "xyz_1" in body_frame_df.columns:
                marker_xyz = body_frame_df[["xyz_1", "xyz_2", "xyz_3"]].values
                sw = body_frame_df[["sw_1", "sw_2", "sw_3"]].values
                ba = body_frame_df[["body_ax_1", "body_ax_2", "body_ax_3"]].values
                up = body_frame_df[["up_1", "up_2", "up_3"]].values

                rotated = rotate_to_body_frame(marker_xyz, sw, ba, up)
                body_frame_df["rot_x"] = rotated[:, 0]
                body_frame_df["rot_y"] = rotated[:, 1]
                body_frame_df["rot_z"] = rotated[:, 2]

                pitch, yaw, roll = extract_body_angles(ba, sw, up)
                body_frame_df["body_pitch"] = pitch
                body_frame_df["body_yaw"] = yaw
                body_frame_df["body_roll"] = roll
        else:
            body_frame_df = pd.DataFrame()
    else:
        body_frame_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Step 16: Label metadata
    # ------------------------------------------------------------------
    if info_df is not None:
        logger.info("Step 16: Labelling metadata")
        for table in [smooth_bp, smooth_tp, smooth_hp]:
            if "seqID" in table.columns and "seqID" in info_df.columns:
                for col in ["Obstacle", "IMU"]:
                    if col in info_df.columns:
                        lookup = info_df.set_index("seqID")[col].to_dict()
                        table[col] = table["seqID"].map(lookup).fillna(0).astype(int)

    # Add flight phase to body_frame_df
    if not body_frame_df.empty and "horzDist" in body_frame_df.columns:
        body_frame_df["flightPhase"] = 0
        for phase, (lower, upper) in config.flight_phases.items():
            mask = (
                (body_frame_df["horzDist"] > lower) &
                (body_frame_df["horzDist"] < upper)
            )
            body_frame_df.loc[mask, "flightPhase"] = phase

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    results = {
        "smooth_backpack": smooth_bp,
        "smooth_tailpack": smooth_tp,
        "smooth_headpack": smooth_hp,
        "labelled": labelled_with_rel,
        "body_frame": body_frame_df,
    }

    logger.info("=" * 60)
    logger.info("Whole-body analysis complete")
    for name, df in results.items():
        logger.info("  %s: %d rows x %d cols", name, len(df), len(df.columns))
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_relative_positions(
    df: pd.DataFrame,
    smooth_bp: pd.DataFrame,
) -> pd.DataFrame:
    """Add relative XYZ columns to a marker table using smooth backpack."""
    if smooth_bp.empty:
        return df.copy()

    return compute_relative_positions(
        df,
        smooth_bp,
        xyz_cols=("X", "Y", "Z"),
        smooth_cols=("smooth_X", "smooth_Y", "smooth_Z"),
        join_col="frameID",
        output_cols=("xyz_1", "xyz_2", "xyz_3"),
    )


def _add_relative_to_smooth(
    smooth_marker: pd.DataFrame,
    smooth_bp: pd.DataFrame,
) -> pd.DataFrame:
    """Add relative XYZ to a smooth marker table using smooth backpack as origin."""
    if smooth_bp.empty or smooth_marker.empty:
        return smooth_marker.copy()

    bp_subset = smooth_bp[
        ["frameID", "smooth_X", "smooth_Y", "smooth_Z"]
    ].drop_duplicates("frameID").rename(columns={
        "smooth_X": "origin_X",
        "smooth_Y": "origin_Y",
        "smooth_Z": "origin_Z",
    })

    merged = smooth_marker.merge(bp_subset, on="frameID", how="inner")
    merged["xyz_1"] = merged["smooth_X"] - merged["origin_X"]
    merged["xyz_2"] = merged["smooth_Y"] - merged["origin_Y"]
    merged["xyz_3"] = merged["smooth_Z"] - merged["origin_Z"]

    merged["horzDist"] = np.sqrt(merged["origin_X"] ** 2 + merged["origin_Y"] ** 2)

    return merged
