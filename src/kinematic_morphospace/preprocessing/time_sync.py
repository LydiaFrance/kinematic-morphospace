"""
Time variable creation for motion-capture recordings.

Detects the takeoff frame from body speed and position criteria, then
computes a time variable (in seconds) relative to that frame.

Reproduces the time-synchronisation logic from ``run_mocap_processing.m``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Takeoff detection
# ---------------------------------------------------------------------------


def find_takeoff_frame(
    body_stats: pd.DataFrame,
    *,
    y_range: tuple[float, float] = (-8.935, -8.5),
    min_speed: float = 2.0,
) -> int | None:
    """Find the first frame where the bird satisfies takeoff criteria.

    The bird must be within the specified Y-position range (near the launch
    perch, ~9 m away) and travelling above the minimum speed.

    Parameters
    ----------
    body_stats : pd.DataFrame
        Per-frame body statistics with columns ``frame``, ``smooth_Y``,
        ``speed``
        (from :func:`~kinematic_morphospace.preprocessing.smoothing.compute_body_statistics`).
    y_range : tuple[float, float]
        (y_min, y_max) range for Y-position at takeoff.
        Default ``(-8.935, -8.5)`` — near the 9 m perch, after origin shift.
    min_speed : float
        Minimum speed (m/s) to confirm active flight (default 2.0).

    Returns
    -------
    int or None
        Frame number of the detected takeoff, or None if no frame matches.
    """
    y_min, y_max = y_range
    mask = (
        (body_stats["smooth_Y"] > y_min)
        & (body_stats["smooth_Y"] < y_max)
        & (body_stats["speed"] > min_speed)
    )

    candidates = body_stats.loc[mask, "frame"]
    if candidates.empty:
        logger.warning(
            "  No frame matches takeoff criteria "
            "(Y in [%.3f, %.3f], speed > %.1f m/s)",
            y_min, y_max, min_speed,
        )
        return None

    frame_zero = int(candidates.min())
    logger.info("  Takeoff frame: %d", frame_zero)
    return frame_zero


# ---------------------------------------------------------------------------
# Time variable
# ---------------------------------------------------------------------------


def create_time_variable(
    df: pd.DataFrame,
    frame_zero: int,
    frame_rate: float,
) -> pd.DataFrame:
    """Add a ``time`` column (in seconds) relative to the takeoff frame.

    Parameters
    ----------
    df : pd.DataFrame
        Table with a ``frame`` column.
    frame_zero : int
        The frame number corresponding to t = 0 (takeoff).
    frame_rate : float
        Recording frame rate in Hz.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``time`` column (seconds).
    """
    df = df.copy()
    df["time"] = (df["frame"] - frame_zero) / frame_rate
    logger.info(
        "  Time range: [%.3f, %.3f] s (frame_zero=%d, rate=%.0f Hz)",
        df["time"].min(), df["time"].max(), frame_zero, frame_rate,
    )
    return df
