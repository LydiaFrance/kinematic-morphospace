"""Coordinate system transformation for hawk flight arena.

Shifts the origin to the target perch, mirrors leftward-approaching flights
so all flights have the same direction convention, and computes horizontal
distance from the perch.

Reproduces the coordinate transform logic from ``run_mocap_processing.m``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arena constants (metres, motion-capture global frame)
# ---------------------------------------------------------------------------

#: Left perch position [X, Y, Z] in the global frame
LEFT_PERCH: np.ndarray = np.array([0.0, -6.658, 0.0])

#: Right perch position [X, Y, Z] in the global frame
RIGHT_PERCH: np.ndarray = np.array([0.0, 2.4238, 0.0])


# ---------------------------------------------------------------------------
# Flight direction
# ---------------------------------------------------------------------------


def detect_flight_direction(
    body_stats: pd.DataFrame,
    *,
    initial_fraction: float = 0.1,
) -> int:
    """Detect whether the bird flies rightward or leftward across the arena.

    Averages the smoothed Y-position over the first ``initial_fraction`` of
    frames to determine which perch the bird started from. This is needed to
    orient all flights consistently (all approaching from negative Y).

    Args:
        body_stats: Per-frame body statistics DataFrame with a ``smooth_Y``
            column, as produced by
            :func:`~kinematic_morphospace.preprocessing.smoothing.compute_body_statistics`.
        initial_fraction: Fraction of initial frames used to compute the mean
            starting Y-position. Defaults to 0.1 (10%, matching MATLAB).

    Returns:
        ``-1`` if the bird starts at negative Y (heading rightward, toward
        the right perch), or ``+1`` if starting at positive Y (heading
        leftward, toward the left perch).
    """
    n_initial = max(1, int(np.floor(len(body_stats) * initial_fraction)))
    mean_y = body_stats["smooth_Y"].iloc[:n_initial].mean()
    direction = int(np.sign(mean_y))

    if direction == 0:
        logger.warning("  Ambiguous flight direction (mean Y ≈ 0), defaulting to -1")
        direction = -1

    if direction == -1:
        label = "rightward (toward right perch)"
    else:
        label = "leftward (toward left perch)"
    logger.info("  Flight direction: %s (sign=%d)", label, direction)
    return direction


# ---------------------------------------------------------------------------
# Origin shift and mirroring
# ---------------------------------------------------------------------------


def shift_origin_to_perch(
    df: pd.DataFrame,
    direction: int,
    *,
    left_perch: np.ndarray | None = None,
    right_perch: np.ndarray | None = None,
    y_column: str = "Y",
) -> pd.DataFrame:
    """Translate the Y coordinate so the target (destination) perch is at the origin.

    Rightward flights (``direction=-1``) subtract the right-perch Y.
    Leftward flights (``direction=+1``) subtract the left-perch Y and negate
    the result, so that all flights approach from large negative Y toward zero
    regardless of the physical direction.

    Args:
        df: Marker or body-stats DataFrame containing the Y-coordinate column
            to shift.
        direction: Flight direction — ``-1`` for rightward, ``+1`` for
            leftward, as returned by :func:`detect_flight_direction`.
        left_perch: Left perch position [X, Y, Z] in metres. Defaults to
            :data:`LEFT_PERCH`.
        right_perch: Right perch position [X, Y, Z] in metres. Defaults to
            :data:`RIGHT_PERCH`.
        y_column: Name of the Y-coordinate column to transform.

    Returns:
        Copy of ``df`` with the Y column shifted (and negated for leftward
        flights).
    """
    lp = left_perch if left_perch is not None else LEFT_PERCH
    rp = right_perch if right_perch is not None else RIGHT_PERCH
    df = df.copy()

    if direction == -1:
        # Rightward: subtract right perch Y
        df[y_column] = df[y_column] - rp[1]
    else:
        # Leftward: subtract left perch Y and negate
        df[y_column] = -(df[y_column] - lp[1])

    logger.info("  Origin shifted to target perch (direction=%d)", direction)
    return df


def shift_origin_all_columns(
    df: pd.DataFrame,
    direction: int,
    *,
    left_perch: np.ndarray | None = None,
    right_perch: np.ndarray | None = None,
    y_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply perch-origin shifting to multiple Y-coordinate columns at once.

    Convenience wrapper around :func:`shift_origin_to_perch` for DataFrames
    that store Y in several columns (e.g. raw and smoothed coordinates).
    Columns not present in ``df`` are silently ignored.

    Args:
        df: Marker table containing the Y-coordinate columns.
        direction: Flight direction — ``-1`` for rightward, ``+1`` for
            leftward.
        left_perch: Left perch position override. Defaults to
            :data:`LEFT_PERCH`.
        right_perch: Right perch position override. Defaults to
            :data:`RIGHT_PERCH`.
        y_columns: List of Y-column names to transform. Defaults to
            ``["Y"]``.

    Returns:
        Copy of ``df`` with all specified Y columns shifted.
    """
    cols = y_columns or ["Y"]
    df = df.copy()

    for col in cols:
        if col in df.columns:
            df = shift_origin_to_perch(
                df, direction,
                left_perch=left_perch,
                right_perch=right_perch,
                y_column=col,
            )
    return df


# ---------------------------------------------------------------------------
# Horizontal distance
# ---------------------------------------------------------------------------


def compute_horizontal_distance(
    body_stats: pd.DataFrame,
    *,
    x_column: str = "smooth_X",
    y_column: str = "smooth_Y",
) -> pd.Series:
    """Compute 2D Euclidean distance from the perch-origin in the horizontal plane.

    Args:
        body_stats: Body statistics DataFrame with smoothed X and Y position
            columns.
        x_column: Name of the smoothed X-position column. Defaults to
            ``"smooth_X"``.
        y_column: Name of the smoothed Y-position column. Defaults to
            ``"smooth_Y"``.

    Returns:
        Series of horizontal distances in metres, named ``"HorzDistance"``.
    """
    horz = np.sqrt(
        body_stats[x_column].to_numpy() ** 2 + body_stats[y_column].to_numpy() ** 2
    )
    return pd.Series(horz, index=body_stats.index, name="HorzDistance")


# ---------------------------------------------------------------------------
# Relative positions
# ---------------------------------------------------------------------------


def compute_relative_positions(
    df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    *,
    xyz_cols: tuple[str, str, str] = ("X", "Y", "Z"),
    smooth_cols: tuple[str, str, str] = ("smooth_X", "smooth_Y", "smooth_Z"),
    join_col: str = "frameID",
    output_cols: tuple[str, str, str] = ("xyz_1", "xyz_2", "xyz_3"),
) -> pd.DataFrame:
    """Compute marker positions relative to the smoothed backpack position.

    Merges the smooth backpack coordinates onto the marker table by
    ``join_col``, then subtracts the smooth backpack XYZ from each marker's
    absolute XYZ. The resulting relative positions are body-centred and
    suitable for downstream rotation. Reproduces MATLAB lines 250, 266-268
    of ``run_whole_body_analysis.m``.

    Args:
        df: Marker table with absolute XYZ coordinates and a frame identifier
            column.
        smooth_df: Smooth backpack table with the same frame identifier column
            and smoothed position columns.
        xyz_cols: Column names for the marker X, Y, Z coordinates in ``df``.
        smooth_cols: Column names for the smooth backpack X, Y, Z coordinates
            in ``smooth_df``.
        join_col: Column name used to merge the two tables. Defaults to
            ``"frameID"``.
        output_cols: Column names for the output relative-position coordinates.

    Returns:
        ``df`` with the relative position columns and merged smooth backpack
        columns added.
    """
    # Select only needed columns from smooth_df to avoid conflicts
    smooth_subset = smooth_df[[join_col, *list(smooth_cols)]].drop_duplicates(  # type: ignore[call-overload]
        subset=[join_col]
    )

    merged = df.merge(smooth_subset, on=join_col, how="inner")

    for out_col, xyz_c, sm_c in zip(output_cols, xyz_cols, smooth_cols, strict=False):
        merged[out_col] = merged[xyz_c] - merged[sm_c]

    logger.info("  Relative positions: %d rows (from %d input)", len(merged), len(df))
    return merged
