"""Position and time calibration for hawk flight data.

Position calibration subtracts perch height from Z coordinates (2020 only).
Time calibration finds the frame at a given horizontal distance from the perch
and sets that as t=0, using progressive tolerance relaxation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position calibration
# ---------------------------------------------------------------------------


def calibrate_position(
    df: pd.DataFrame,
    perch_height: float = 1.25,
    z_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Subtract the perch height from Z-coordinate columns to zero the vertical datum.

    Only columns present in ``df`` are modified. Multi-column fields use the
    ``_3`` suffix convention for the Z component (e.g. ``XYZ_3``,
    ``smooth_XYZ_3``, ``backpack_smooth_XYZ_3``). The 2020 dataset was
    recorded with the perch at 1.25 m above the motion-capture floor.

    Args:
        df: DataFrame containing Z-coordinate columns to adjust.
        perch_height: Height of the perch in metres, subtracted from all Z
            values. Defaults to 1.25.
        z_columns: Explicit list of Z-column names to adjust. Defaults to
            ``["XYZ_3", "smooth_XYZ_3", "backpack_smooth_XYZ_3"]``.

    Returns:
        Copy of ``df`` with the perch height subtracted from each Z column.
    """
    df = df.copy()

    if z_columns is None:
        z_columns = ["XYZ_3", "smooth_XYZ_3", "backpack_smooth_XYZ_3"]

    adjusted = []
    for col in z_columns:
        if col in df.columns:
            df[col] = df[col] - perch_height
            adjusted.append(col)

    if adjusted:
        logger.info(
            "  Position calibrated (perch_height=%.2f) on: %s",
            perch_height,
            ", ".join(adjusted),
        )
    else:
        logger.warning("  No Z columns found to calibrate")

    return df


# ---------------------------------------------------------------------------
# Time calibration
# ---------------------------------------------------------------------------


def find_jump_frame(
    seq_df: pd.DataFrame,
    jump_dist: float = 8.3,
    tolerances: tuple[float, ...] = (0.02, 0.05, 0.2),
    distance_col: str = "HorzDistance",
    time_col: str = "time",
) -> float:
    """Find the time at which a flight sequence crosses a target horizontal distance.

    Uses progressive tolerance relaxation: tries each tolerance in order and
    returns the time of the closest matching frame. At the widest tolerance,
    takes the mean time of all matching frames to replicate the MATLAB
    interpolation behaviour.

    Args:
        seq_df: Single-sequence DataFrame with horizontal distance and time
            columns.
        jump_dist: Target horizontal distance in metres at which to find the
            crossing frame. Defaults to 8.3.
        tolerances: Progressive tolerance values (metres) used for the
            distance match, tried in order from tightest to loosest.
        distance_col: Name of the column containing horizontal distance
            values.
        time_col: Name of the column containing time values.

    Returns:
        Time in seconds at the jump frame, or ``NaN`` if no matching frame
        is found within any tolerance.
    """
    distances = seq_df[distance_col].values
    times = seq_df[time_col].values

    for i, tol in enumerate(tolerances):
        mask = np.abs(distances - jump_dist) <= tol
        if mask.any():
            if i == len(tolerances) - 1:
                # Widest tolerance: take mean time (MATLAB interpolation)
                return float(np.mean(times[mask]))
            # Narrower tolerances: take the closest match
            idx = np.argmin(np.abs(distances - jump_dist))
            return float(times[idx])

    return float("nan")


def calibrate_time(
    df: pd.DataFrame,
    jump_dist: float = 8.3,
    tolerances: tuple[float, ...] = (0.02, 0.05, 0.2),
    seq_col: str = "seqID",
    distance_col: str = "HorzDistance",
    time_col: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set t=0 at the jump-distance crossing frame for each flight sequence.

    For each unique sequence, uses :func:`find_jump_frame` to locate the
    frame where the bird crosses ``jump_dist`` metres of horizontal distance,
    then subtracts that time from all frames in the sequence. The resulting
    offset table can be applied to other tables (e.g. labelled markers) via
    :func:`apply_time_offsets`.

    Args:
        df: DataFrame with sequence identifier, horizontal distance, and time
            columns.
        jump_dist: Horizontal distance (metres) at which t=0 is defined.
            Defaults to 8.3.
        tolerances: Progressive tolerance values passed to
            :func:`find_jump_frame`.
        seq_col: Column name identifying individual flight sequences.
        distance_col: Column name for horizontal distance values.
        time_col: Column name for time values to calibrate.

    Returns:
        Tuple of (calibrated_df, offset_df). ``calibrated_df`` is a copy of
        ``df`` with adjusted time values. ``offset_df`` has columns
        ``seqID`` and ``time_offset`` (the subtracted values), suitable for
        passing to :func:`apply_time_offsets`.
    """
    df = df.copy()
    sequences = df[seq_col].unique()
    offsets = []

    n_matched = 0
    n_failed = 0

    for seq in sequences:
        seq_mask = df[seq_col] == seq
        seq_data = df.loc[seq_mask]

        t0 = find_jump_frame(
            seq_data,
            jump_dist=jump_dist,
            tolerances=tolerances,
            distance_col=distance_col,
            time_col=time_col,
        )

        offsets.append({"seqID": seq, "time_offset": t0})

        if np.isnan(t0):
            n_failed += 1
            logger.warning("  No jump frame found for sequence %s", seq)
        else:
            df.loc[seq_mask, time_col] = df.loc[seq_mask, time_col] - t0
            n_matched += 1

    offset_df = pd.DataFrame(offsets)
    logger.info(
        "  Time calibrated: %d/%d sequences (jump_dist=%.1f)",
        n_matched,
        len(sequences),
        jump_dist,
    )
    if n_failed:
        logger.warning("  %d sequences had no jump-frame match", n_failed)

    return df, offset_df


def apply_time_offsets(
    df: pd.DataFrame,
    offset_df: pd.DataFrame,
    seq_col: str = "seqID",
    time_col: str = "time",
) -> pd.DataFrame:
    """Apply pre-computed time offsets from :func:`calibrate_time` to another table.

    Used to propagate the trajectory-derived t=0 calibration to other tables
    such as the labelled marker or body-marker tables, ensuring consistent
    time reference across all outputs.

    Args:
        df: Table whose time column should be adjusted.
        offset_df: Lookup table with ``seqID`` and ``time_offset`` columns,
            as returned by :func:`calibrate_time`.
        seq_col: Column name identifying flight sequences.
        time_col: Column name for time values to adjust.

    Returns:
        Copy of ``df`` with the per-sequence time offsets subtracted.
    """
    df = df.copy()
    offset_map = dict(zip(offset_df["seqID"], offset_df["time_offset"], strict=False))

    for seq, t0 in offset_map.items():
        if np.isnan(t0):
            continue
        mask = df[seq_col] == seq
        if mask.any():
            df.loc[mask, time_col] = df.loc[mask, time_col] - t0

    return df
