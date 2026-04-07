"""Trajectory smoothing and derivative computation.

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
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Moving mean smoothing
# ---------------------------------------------------------------------------


def moving_mean_smooth(
    values: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Apply a centred moving-average filter, matching MATLAB ``movmean``.

    Args:
        values: 1D array of values to smooth.
        window: Number of samples in the smoothing window. Defaults to 10.

    Returns:
        Smoothed array of the same length as ``values``.
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
    """Fit a smoothing cubic spline and return values, velocity, and acceleration.

    Uses ``scipy.interpolate.UnivariateSpline`` with a smoothing factor
    derived from ``rms`` as ``s = rms**2 * len(x)``. This approximates
    MATLAB's ``spaps`` function with a different but analogous parameterisation.

    Args:
        x: Independent variable array, e.g. time or frame numbers.
        y: Dependent variable array, e.g. one spatial coordinate.
        rms: Desired RMS residual tolerance. Converted to a scipy smoothing
            factor via ``s = rms**2 * len(x)``. Defaults to 0.0001.

    Returns:
        Tuple of three arrays ``(y_smooth, velocity, acceleration)``.
        ``y_smooth`` is the spline-smoothed signal. ``velocity`` is the
        first derivative (dy/dx). ``acceleration`` is the second derivative
        (d²y/dx²).

    Raises:
        ImportError: If scipy is not installed.
    """
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
    """Smooth a 3D marker trajectory, excising large gap regions before fitting.

    Reproduces the per-sequence smoothing logic from MATLAB
    ``run_whole_body_analysis.m`` (steps 1 and 8):

    1. Reconstruct the full frame/time signal from sparse observations.
    2. Detect inter-observation gaps larger than ``max_gap_frames``.
    3. Skip gaps that occur before ``min_time`` or within ``min_horz_dist``
       of the perch (typically pre-flight).
    4. Remove large gap regions from the interpolation grid.
    5. Fit a smoothing spline per coordinate over the gap-free grid.

    Args:
        time: (M,) observed time values in seconds.
        frames: (M,) observed frame numbers (integers).
        xyz: (M, 3) observed marker positions in metres.
        rms: Smoothing tolerance passed to :func:`smooth_spline`. Defaults
            to 0.001.
        frame_rate: Recording frame rate in Hz, used to reconstruct the
            time grid. Defaults to 200.0.
        max_gap_frames: Gaps larger than this number of frames are removed
            from the interpolation grid. Defaults to 30.
        min_time: Gaps occurring before this time (seconds) are ignored.
            Defaults to 0.0.
        min_horz_dist: Gaps occurring within this horizontal distance
            (metres) from the perch are ignored. Defaults to 0.3.
        horz_dist: (M,) horizontal distance from the perch for each
            observation. If None, distance-based gap filtering is skipped.

    Returns:
        Dict with keys ``"frames"`` (K,), ``"time"`` (K,), ``"smooth"``
        (K, 3), ``"velocity"`` (K, 3), ``"acceleration"`` (K, 3), and
        ``"gaps"`` (list of dicts with ``frame``, ``size``, ``time``).
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
        _s, _v, _a = smooth_spline(time, xyz[:, coord], rms=rms)
        # Re-evaluate at the output time signal
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
    """Compute per-frame body position, smoothed trajectory, velocity, and speed.

    Groups body-pack markers (backpack, tailpack, headpack) by frame, computes
    the mean position, applies moving-average smoothing, then estimates
    velocity via numerical gradient and speed as the smoothed norm of velocity.

    Args:
        df: Marker table with columns ``frame``, ``marker_id``, ``X``,
            ``Y``, ``Z``.
        body_labels: Series indexed by ``marker_id`` with body-pack labels.
            If provided, only markers labelled ``"backpack"``,
            ``"tailpack"``, or ``"headpack"`` are included. If None, all
            markers are used.
        smooth_window: Moving-mean window size in frames for position
            smoothing. Defaults to 10 (matching MATLAB).
        frame_rate: Recording frame rate in Hz, used to scale velocity from
            frames to seconds. Defaults to 200.0.

    Returns:
        DataFrame with one row per frame and columns ``frame``,
        ``mean_X/Y/Z``, ``smooth_X/Y/Z``, ``vel_X/Y/Z``, and ``speed``
        (m/s).
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
    smooth_x_vals = frame_mean["mean_X"].values
    smooth_y_vals = frame_mean["mean_Y"].values
    smooth_z_vals = frame_mean["mean_Z"].values
    frame_mean["smooth_X"] = moving_mean_smooth(smooth_x_vals, smooth_window)
    frame_mean["smooth_Y"] = moving_mean_smooth(smooth_y_vals, smooth_window)
    frame_mean["smooth_Z"] = moving_mean_smooth(smooth_z_vals, smooth_window)

    # Velocity: gradient of smoothed position, scaled by frame rate
    vel_x_vals = frame_mean["smooth_X"].values
    vel_y_vals = frame_mean["smooth_Y"].values
    vel_z_vals = frame_mean["smooth_Z"].values
    frame_mean["vel_X"] = (
        np.gradient(vel_x_vals, frames) * frame_rate
    )
    frame_mean["vel_Y"] = (
        np.gradient(vel_y_vals, frames) * frame_rate
    )
    frame_mean["vel_Z"] = (
        np.gradient(vel_z_vals, frames) * frame_rate
    )

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
