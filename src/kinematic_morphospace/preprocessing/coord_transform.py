"""
Coordinate system transformation for hawk flight arena.

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
    """Detect whether the bird flies leftward or rightward.

    Uses the mean Y-position over the first ``initial_fraction`` of frames
    to determine the starting side.

    Parameters
    ----------
    body_stats : pd.DataFrame
        Per-frame body statistics with ``smooth_Y`` column
        (from :func:`~kinematic_morphospace.preprocessing.smoothing.compute_body_statistics`).
    initial_fraction : float
        Fraction of initial frames to average (default 10%, matching MATLAB).

    Returns
    -------
    int
        ``-1`` if the bird starts at negative Y (heading rightward, toward
        the right perch) or ``+1`` if starting at positive Y (heading
        leftward, toward the left perch).
    """
    n_initial = max(1, int(np.floor(len(body_stats) * initial_fraction)))
    mean_y = body_stats["smooth_Y"].iloc[:n_initial].mean()
    direction = int(np.sign(mean_y))

    if direction == 0:
        logger.warning("  Ambiguous flight direction (mean Y ≈ 0), defaulting to -1")
        direction = -1

    label = "rightward (toward right perch)" if direction == -1 else "leftward (toward left perch)"
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
    """Translate coordinates so the target perch is at the Y-origin.

    For rightward flights (``direction=-1``), subtracts the right perch Y.
    For leftward flights (``direction=+1``), subtracts the left perch Y
    and negates Y so all flights approach from negative toward zero.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with a Y-coordinate column.
    direction : int
        Flight direction: ``-1`` (rightward) or ``+1`` (leftward).
    left_perch : np.ndarray, optional
        Override left perch position. Default :data:`LEFT_PERCH`.
    right_perch : np.ndarray, optional
        Override right perch position. Default :data:`RIGHT_PERCH`.
    y_column : str
        Name of the Y-coordinate column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with Y shifted (and possibly negated).
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
    """Apply :func:`shift_origin_to_perch` to multiple Y-coordinate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table.
    direction : int
        Flight direction.
    left_perch, right_perch : np.ndarray, optional
        Override perch positions.
    y_columns : list[str], optional
        Y-coordinate column names to transform. Defaults to ``["Y"]``.

    Returns
    -------
    pd.DataFrame
        Copy with all specified Y columns shifted.
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
    """Compute 2-D horizontal distance from the perch origin.

    Parameters
    ----------
    body_stats : pd.DataFrame
        Body statistics with smoothed X and Y columns.
    x_column : str
        Name of the X column (default ``"smooth_X"``).
    y_column : str
        Name of the Y column (default ``"smooth_Y"``).

    Returns
    -------
    pd.Series
        Horizontal distance (metres), named ``"HorzDistance"``.
    """
    horz = np.sqrt(
        body_stats[x_column].values ** 2 + body_stats[y_column].values ** 2
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
    """Compute marker positions relative to the smooth backpack position.

    Merges the smooth backpack coordinates onto the marker table by
    ``join_col``, then subtracts smooth XYZ from marker XYZ.

    Reproduces MATLAB lines 250, 266-268 of ``run_whole_body_analysis.m``.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with absolute XYZ coordinates and ``join_col``.
    smooth_df : pd.DataFrame
        Smooth backpack table with ``join_col`` and smooth coordinate columns.
    xyz_cols : tuple of str
        Column names for marker X, Y, Z in *df*.
    smooth_cols : tuple of str
        Column names for smooth X, Y, Z in *smooth_df*.
    join_col : str
        Column to merge on (default ``"frameID"``).
    output_cols : tuple of str
        Column names for the output relative coordinates.

    Returns
    -------
    pd.DataFrame
        *df* with added relative position columns and merged smooth columns.
    """
    # Select only needed columns from smooth_df to avoid conflicts
    smooth_subset = smooth_df[[join_col] + list(smooth_cols)].drop_duplicates(
        subset=[join_col]
    )

    merged = df.merge(smooth_subset, on=join_col, how="inner")

    for out_col, xyz_c, sm_c in zip(output_cols, xyz_cols, smooth_cols):
        merged[out_col] = merged[xyz_c] - merged[sm_c]

    logger.info("  Relative positions: %d rows (from %d input)", len(merged), len(df))
    return merged
