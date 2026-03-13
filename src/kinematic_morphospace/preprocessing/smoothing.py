"""
Trajectory smoothing and derivative computation.

Provides moving-average smoothing and optional spline fitting for marker
trajectories. Also computes per-frame body statistics (mean body position,
velocity, speed) from the labelled body markers.

Reproduces the smoothing and body-statistics logic from
``run_mocap_processing.m``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Moving mean smoothing
# ---------------------------------------------------------------------------


def moving_mean_smooth(
    values: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Apply a centred moving average, matching MATLAB ``movmean``.

    Parameters
    ----------
    values : np.ndarray
        1-D array of values to smooth.
    window : int
        Smoothing window size.

    Returns
    -------
    np.ndarray
        Smoothed values (same length as input).
    """
    if window < 1:
        return values.copy()
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - pad_left - 1
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Spline smoothing (optional, scipy-based)
# ---------------------------------------------------------------------------


def smooth_spline(
    x: np.ndarray,
    y: np.ndarray,
    rms: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a smoothing cubic spline and compute velocity + acceleration.

    Uses ``scipy.interpolate.UnivariateSpline`` with a smoothing factor
    derived from *rms*. This is an approximation of MATLAB's ``spaps``
    function with different parameterisation.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g. frame numbers).
    y : np.ndarray
        Dependent variable (e.g. one coordinate).
    rms : float
        Desired RMS smoothing tolerance. Converted to ``s = rms**2 * len(x)``
        for UnivariateSpline.

    Returns
    -------
    y_smooth : np.ndarray
        Smoothed values.
    velocity : np.ndarray
        First derivative (dy/dx).
    acceleration : np.ndarray
        Second derivative (d²y/dx²).

    Raises
    ------
    ImportError
        If scipy is not installed.
    """
    from scipy.interpolate import UnivariateSpline

    s = rms**2 * len(x)
    spline = UnivariateSpline(x, y, s=s, k=3)

    y_smooth = spline(x)
    velocity = spline.derivative(n=1)(x)
    acceleration = spline.derivative(n=2)(x)

    return y_smooth, velocity, acceleration


# ---------------------------------------------------------------------------
# Gap-aware trajectory smoothing
# ---------------------------------------------------------------------------


def smooth_trajectory_with_gaps(
    time: np.ndarray,
    frames: np.ndarray,
    xyz: np.ndarray,
    *,
    rms: float = 0.001,
    frame_rate: float = 200.0,
    max_gap_frames: int = 30,
    min_time: float = 0.0,
    min_horz_dist: float = 0.3,
    horz_dist: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Smooth a marker trajectory with gap detection and removal.

    Reproduces the per-sequence smoothing logic from MATLAB
    ``run_whole_body_analysis.m`` (steps 1 and 8):

    1. Reconstruct full frame/time signals from sparse observations.
    2. Detect gaps > ``max_gap_frames`` frames.
    3. Exclude gap regions that occur before ``min_time`` or closer than
       ``min_horz_dist`` to the perch.
    4. Remove large gap regions from the interpolation signal.
    5. Apply :func:`smooth_spline` per coordinate.

    Parameters
    ----------
    time : np.ndarray
        (M,) observed time values.
    frames : np.ndarray
        (M,) observed frame numbers (integers).
    xyz : np.ndarray
        (M, 3) observed marker positions.
    rms : float
        Smoothing tolerance for the spline fit.
    frame_rate : float
        Recording frame rate in Hz.
    max_gap_frames : int
        Gaps larger than this are flagged and removed from interpolation.
    min_time : float
        Ignore gaps occurring before this time (e.g. before takeoff).
    min_horz_dist : float
        Ignore gaps occurring closer than this horizontal distance to perch.
    horz_dist : np.ndarray, optional
        (M,) horizontal distance from perch for each observation.
        If None, distance-based gap filtering is skipped.

    Returns
    -------
    dict
        ``"frames"``: (K,) output frame numbers,
        ``"time"``: (K,) output time values,
        ``"smooth"``: (K, 3) smoothed positions,
        ``"velocity"``: (K, 3) velocity,
        ``"acceleration"``: (K, 3) acceleration,
        ``"gaps"``: list of dicts with gap info.
    """
    frames = np.asarray(frames, dtype=int)
    time = np.asarray(time, dtype=float)
    xyz = np.asarray(xyz, dtype=float)

    # 1. Full frame and time signals
    frame_signal = np.arange(frames.min(), frames.max() + 1)
    sample_rate = 1.0 / frame_rate
    time_signal = np.arange(time.min(), time.max() + sample_rate / 2, sample_rate)

    # Ensure frame and time signals are the same length (use shorter)
    n = min(len(frame_signal), len(time_signal))
    frame_signal = frame_signal[:n]
    time_signal = time_signal[:n]

    # 2. Detect gaps
    frame_diffs = np.diff(frames)
    gap_mask = frame_diffs > 1

    gaps = []
    if gap_mask.any():
        gap_indices = np.where(gap_mask)[0]
        for idx in gap_indices:
            gap_size = int(frame_diffs[idx])
            gap_frame = int(frames[idx])
            gap_time = float(time[idx])
            gap_hdist = float(horz_dist[idx]) if horz_dist is not None else np.inf

            # Skip small gaps
            if gap_size <= max_gap_frames:
                continue

            # Skip gaps before min_time or near perch
            if gap_time < min_time and gap_size > 1:
                continue
            if gap_hdist < min_horz_dist and gap_size > 1:
                continue

            gaps.append({
                "frame": gap_frame,
                "size": gap_size,
                "time": gap_time,
            })

    # 3. Remove large gap regions from interpolation signals
    for gap in gaps:
        gap_start = gap["frame"]
        gap_end = gap_start + gap["size"]
        keep = ~((frame_signal > gap_start) & (frame_signal < gap_end))
        frame_signal = frame_signal[keep]
        time_signal = time_signal[keep]

    # 4. Spline smooth per coordinate
    n_out = len(time_signal)
    smooth = np.empty((n_out, 3))
    velocity = np.empty((n_out, 3))
    acceleration = np.empty((n_out, 3))

    for coord in range(3):
        s, v, a = smooth_spline(time, xyz[:, coord], rms=rms)
        # Re-evaluate at the output time signal
        from scipy.interpolate import UnivariateSpline
        s_param = rms**2 * len(time)
        spline = UnivariateSpline(time, xyz[:, coord], s=s_param, k=3)
        smooth[:, coord] = spline(time_signal)
        velocity[:, coord] = spline.derivative(n=1)(time_signal)
        acceleration[:, coord] = spline.derivative(n=2)(time_signal)

    return {
        "frames": frame_signal,
        "time": time_signal,
        "smooth": smooth,
        "velocity": velocity,
        "acceleration": acceleration,
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Body statistics
# ---------------------------------------------------------------------------


def compute_body_statistics(
    df: pd.DataFrame,
    body_labels: pd.Series | None = None,
    *,
    smooth_window: int = 10,
    frame_rate: float = 200.0,
) -> pd.DataFrame:
    """Compute per-frame body position, smoothed XYZ, velocity, and speed.

    Groups body markers (backpack + tailpack + headpack) by frame, computes
    the mean position, applies moving-average smoothing, then numerical
    gradient for velocity and speed.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with ``frame``, ``marker_id``, ``X``, ``Y``, ``Z``.
    body_labels : pd.Series, optional
        Series indexed by ``marker_id`` with body-pack labels. If provided,
        only markers labelled ``"backpack"``, ``"tailpack"``, or
        ``"headpack"`` are included. If None, all markers are used.
    smooth_window : int
        Moving-mean window for position smoothing (default 10 frames,
        matching MATLAB).
    frame_rate : float
        Recording frame rate in Hz (default 200).

    Returns
    -------
    pd.DataFrame
        One row per frame with columns:
        ``frame``, ``mean_X``, ``mean_Y``, ``mean_Z``,
        ``smooth_X``, ``smooth_Y``, ``smooth_Z``,
        ``vel_X``, ``vel_Y``, ``vel_Z``, ``speed``.
    """
    # Filter to body markers if labels provided
    if body_labels is not None:
        body_ids = body_labels[
            body_labels.isin(["backpack", "tailpack", "headpack"])
        ].index
        body = df[df["marker_id"].isin(body_ids)].copy()
    else:
        body = df.copy()

    # Per-frame mean position
    frame_mean = body.groupby("frame")[["X", "Y", "Z"]].mean().sort_index()
    frame_mean.columns = ["mean_X", "mean_Y", "mean_Z"]

    frames = frame_mean.index.values.astype(float)

    # Smooth XYZ with moving mean
    frame_mean["smooth_X"] = moving_mean_smooth(frame_mean["mean_X"].values, smooth_window)
    frame_mean["smooth_Y"] = moving_mean_smooth(frame_mean["mean_Y"].values, smooth_window)
    frame_mean["smooth_Z"] = moving_mean_smooth(frame_mean["mean_Z"].values, smooth_window)

    # Velocity: gradient of smoothed position, scaled by frame rate
    frame_mean["vel_X"] = np.gradient(frame_mean["smooth_X"].values, frames) * frame_rate
    frame_mean["vel_Y"] = np.gradient(frame_mean["smooth_Y"].values, frames) * frame_rate
    frame_mean["vel_Z"] = np.gradient(frame_mean["smooth_Z"].values, frames) * frame_rate

    # Speed: norm of velocity, with additional smoothing
    velocity = frame_mean[["vel_X", "vel_Y", "vel_Z"]].values
    raw_speed = np.linalg.norm(velocity, axis=1)
    speed_window = max(1, int(frame_rate / 10))
    frame_mean["speed"] = moving_mean_smooth(raw_speed, speed_window)

    result = frame_mean.reset_index()
    logger.info(
        "  Body stats: %d frames, speed range [%.2f, %.2f] m/s",
        len(result), result["speed"].min(), result["speed"].max(),
    )
    return result
