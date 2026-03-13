"""
Flight segmentation from continuous motion-capture recordings.

Detects individual flights from the velocity profile of moving markers,
using peak detection on the absolute gradient of the smoothed median
Y-position. Trial boundaries are stored as JSON annotations for
reproducibility.

Reproduces the trial-splitting logic from ``run_mocap_processing.m``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Velocity peak detection
# ---------------------------------------------------------------------------


def detect_velocity_peaks(
    df: pd.DataFrame,
    *,
    min_peak_distance: int = 250,
    min_peak_width: int = 150,
    min_peak_height: float = 0.01,
    smooth_fraction: float = 0.05,
) -> pd.DataFrame:
    """Auto-detect flights from median Y velocity peaks.

    Reproduces the MATLAB ``findpeaks`` call on
    ``abs(gradient(movmean(median_Y, window)))``.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format marker table with columns ``frame``, ``Y``, and
        ``label_stationary`` (or only moving markers).
    min_peak_distance : int
        Minimum frames between peaks (MATLAB ``MinPeakDistance``).
    min_peak_width : int
        Minimum width of each peak in frames (MATLAB ``MinPeakWidth``).
    min_peak_height : float
        Minimum peak height in velocity units (MATLAB ``MinPeakHeight``).
    smooth_fraction : float
        Smoothing window as fraction of total frames (default 5%).

    Returns
    -------
    pd.DataFrame
        Table with columns ``peak_frame``, ``peak_height``, ``width``,
        ``start_frame``, ``end_frame`` for each detected flight.
    """
    from scipy.signal import find_peaks, peak_widths

    # Filter to moving markers if label_stationary column exists
    if "label_stationary" in df.columns:
        moving = df[~df["label_stationary"]].copy()
    else:
        moving = df.copy()

    # Compute per-frame median Y
    median_y = moving.groupby("frame")["Y"].median().sort_index()
    frames = median_y.index.values.astype(float)
    y_values = median_y.values

    # Smooth with moving mean (window = smooth_fraction * n_frames)
    n_frames = len(y_values)
    window = max(3, int(np.floor(n_frames * smooth_fraction)))
    smoothed = _moving_mean(y_values, window)

    # Compute absolute velocity gradient
    velocity = np.gradient(smoothed, frames)
    abs_velocity = np.abs(velocity)

    # Find peaks
    peaks, properties = find_peaks(
        abs_velocity,
        distance=min_peak_distance,
        height=min_peak_height,
        width=min_peak_width,
    )

    if len(peaks) == 0:
        logger.warning("  No velocity peaks detected")
        return pd.DataFrame(
            columns=["peak_frame", "peak_height", "width", "start_frame", "end_frame"]
        )

    # Get peak widths for trial boundary calculation
    widths, _, _, _ = peak_widths(abs_velocity, peaks, rel_height=0.5)

    # Compute trial boundaries: peak ± 1.5 * width
    frame_min = frames.min()
    frame_max = frames.max()
    frame_range = frame_max - frame_min
    edge_margin = frame_range * 0.05

    results = []
    for i, peak_idx in enumerate(peaks):
        peak_frame = frames[peak_idx]
        w = widths[i]
        start = peak_frame - 1.5 * w
        end = peak_frame + 1.5 * w

        # Edge clamping (within 5% of range edge → clamp to edge)
        if start < frame_min + edge_margin:
            start = frame_min
        if end > frame_max - edge_margin:
            end = frame_max

        results.append({
            "peak_frame": peak_frame,
            "peak_height": abs_velocity[peak_idx],
            "width": w,
            "start_frame": start,
            "end_frame": end,
        })

    result_df = pd.DataFrame(results)
    logger.info("  Detected %d flights", len(result_df))
    return result_df


def _moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, matching MATLAB ``movmean`` behaviour."""
    kernel = np.ones(window) / window
    # Pad to handle edges like MATLAB's movmean
    pad_left = window // 2
    pad_right = window - pad_left - 1
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------


def load_annotations(path: str | Path) -> list[dict]:
    """Load trial boundary annotations from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file with trial annotations.

    Returns
    -------
    list[dict]
        List of dicts with ``start_frame`` and ``end_frame`` keys.
    """
    path = Path(path)
    with path.open() as f:
        annotations = json.load(f)
    logger.info("  Loaded %d trial annotations from %s", len(annotations), path.name)
    return annotations


def save_annotations(annotations: list[dict], path: str | Path) -> None:
    """Save trial boundary annotations to a JSON file.

    Parameters
    ----------
    annotations : list[dict]
        List of dicts with ``start_frame`` and ``end_frame`` keys.
    path : str or Path
        Output path.
    """
    path = Path(path)
    with path.open("w") as f:
        json.dump(annotations, f, indent=2)
    logger.info("  Saved %d trial annotations to %s", len(annotations), path.name)


# ---------------------------------------------------------------------------
# Trial assignment
# ---------------------------------------------------------------------------


def split_by_trial(
    df: pd.DataFrame,
    annotations: list[dict],
) -> pd.DataFrame:
    """Assign a ``trial`` column to each row based on trial boundary annotations.

    Frames not falling within any trial boundary get ``trial=0``.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with a ``frame`` column.
    annotations : list[dict]
        Trial boundaries, each with ``start_frame`` and ``end_frame``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``trial`` column (1-indexed, 0 = no trial).
    """
    df = df.copy()
    df["trial"] = 0

    for i, ann in enumerate(annotations, start=1):
        start = ann["start_frame"]
        end = ann["end_frame"]
        mask = (df["frame"] >= start) & (df["frame"] <= end)
        df.loc[mask, "trial"] = i

    assigned = (df["trial"] > 0).sum()
    logger.info(
        "  Assigned %d rows to %d trials (%d unassigned)",
        assigned, len(annotations), len(df) - assigned,
    )
    return df
