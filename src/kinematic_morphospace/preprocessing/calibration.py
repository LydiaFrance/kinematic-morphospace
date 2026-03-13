"""
Position and time calibration for hawk flight data.

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
    """Subtract perch height from Z-coordinate columns.

    Only columns that exist in *df* are modified. Multi-column fields
    use the ``_3`` suffix convention for the Z component (e.g. ``XYZ_3``,
    ``smooth_XYZ_3``, ``backpack_smooth_XYZ_3``).

    Parameters
    ----------
    df : pd.DataFrame
        Table with Z-coordinate columns.
    perch_height : float
        Height to subtract from Z values, in metres.
    z_columns : list[str], optional
        Explicit list of Z-column names to adjust. If ``None``, defaults
        to ``["XYZ_3", "smooth_XYZ_3", "backpack_smooth_XYZ_3"]``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with adjusted Z columns.
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
    """Find the time at which a sequence crosses *jump_dist* horizontal distance.

    Uses progressive tolerance relaxation: tries each tolerance in order,
    returning the time of the closest matching frame. At the widest tolerance,
    takes the mean time of all matching frames (replicating the MATLAB
    interpolation behaviour).

    Parameters
    ----------
    seq_df : pd.DataFrame
        Single-sequence data with *distance_col* and *time_col*.
    jump_dist : float
        Target horizontal distance in metres.
    tolerances : tuple of float
        Progressive tolerance values for the distance match.
    distance_col : str
        Column containing horizontal distance.
    time_col : str
        Column containing time values.

    Returns
    -------
    float
        Time at the jump frame, or ``NaN`` if no match found.
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
    """Calibrate time so that t=0 is at the jump-distance frame.

    For each unique sequence, finds the frame closest to *jump_dist*
    horizontal distance and subtracts that time from all frames in the
    sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Table with *seq_col*, *distance_col*, and *time_col* columns.
    jump_dist : float
        Target horizontal distance for t=0, in metres.
    tolerances : tuple of float
        Progressive tolerance values.
    seq_col : str
        Column identifying sequences.
    distance_col : str
        Column containing horizontal distance.
    time_col : str
        Column containing time values.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - The calibrated DataFrame (copy of *df* with adjusted time).
        - A lookup table with columns ``seqID`` and ``time_offset``
          (the *newstart* values), useful for applying the same offsets
          to other tables (e.g. labelled markers).
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
    """Apply pre-computed time offsets to another table.

    This is used to apply the trajectory-derived time calibration to the
    labelled and body marker tables.

    Parameters
    ----------
    df : pd.DataFrame
        Table to calibrate.
    offset_df : pd.DataFrame
        Lookup table with ``seqID`` and ``time_offset`` columns
        (from :func:`calibrate_time`).
    seq_col : str
        Sequence column name.
    time_col : str
        Time column name.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with calibrated time.
    """
    df = df.copy()
    offset_map = dict(zip(offset_df["seqID"], offset_df["time_offset"]))

    for seq, t0 in offset_map.items():
        if np.isnan(t0):
            continue
        mask = df[seq_col] == seq
        if mask.any():
            df.loc[mask, time_col] = df.loc[mask, time_col] - t0

    return df
