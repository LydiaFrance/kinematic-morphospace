"""Time variable creation for motion-capture recordings.

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
    """Locate the takeoff frame where the bird first leaves the launch perch.

    Searches for the first frame where the bird is within the expected
    Y-position range near the launch perch and is already travelling above
    the minimum speed threshold, confirming active flight has begun.

    Args:
        body_stats: Per-frame body statistics with columns ``frame``,
            ``smooth_Y``, and ``speed``, as produced by
            :func:`~kinematic_morphospace.preprocessing.smoothing.compute_body_statistics`.
        y_range: (y_min, y_max) Y-position window in metres used to locate
            the launch perch region. Default ``(-8.935, -8.5)`` corresponds
            to the 9 m perch after origin shift.
        min_speed: Minimum body speed in m/s required to confirm active
            flight. Defaults to 2.0.

    Returns:
        Frame number of the detected takeoff, or None if no frame satisfies
        both criteria.
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
    """Add a ``time`` column in seconds, with t=0 at the takeoff frame.

    Args:
        df: DataFrame containing a ``frame`` column with integer frame
            numbers.
        frame_zero: Frame number corresponding to t=0 (the detected takeoff
            frame from :func:`find_takeoff_frame`).
        frame_rate: Recording frame rate in Hz, used to convert frame offsets
            to seconds.

    Returns:
        Copy of ``df`` with a new ``time`` column in seconds.
    """
    df = df.copy()
    df["time"] = (df["frame"] - frame_zero) / frame_rate
    logger.info(
        "  Time range: [%.3f, %.3f] s (frame_zero=%d, rate=%.0f Hz)",
        df["time"].min(), df["time"].max(), frame_zero, frame_rate,
    )
    return df
