"""
Vectorized body-frame rotation and angle computation.

Provides batch computation of pitch/yaw angles, rotation matrix construction,
and body-frame coordinate transformation. Replaces the row-by-row MATLAB
functions ``find_pitchangle.m``, ``find_yawangle.m``, ``rotmat_for_table.m``,
and ``rotate_byRow.m`` with vectorized NumPy equivalents.

Reproduces steps 9, 11-15 of ``run_whole_body_analysis.m``.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_pitch_angle(vectors: np.ndarray) -> np.ndarray:
    """Compute pitch angle from direction vectors.

    Vectorized version of MATLAB ``find_pitchangle.m``: zeros the X component,
    normalizes, then computes ``atan2(dot(v, [0,0,-1]), dot(v, [0,-1,0]))``.

    Parameters
    ----------
    vectors : np.ndarray
        (N, 3) array of direction vectors (e.g. tailpack relative to backpack).

    Returns
    -------
    np.ndarray
        (N,) pitch angles in degrees.
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    v = vectors.copy()
    v[:, 0] = 0.0  # remove lateral (X) component

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    v = v / norms

    # floor_h = [0, -1, 0], floor_v = [0, 0, -1]
    dot_v = -v[:, 2]  # dot(v, [0, 0, -1])
    dot_h = -v[:, 1]  # dot(v, [0, -1, 0])

    return np.degrees(np.arctan2(dot_v, dot_h))


def compute_yaw_angle(vectors: np.ndarray) -> np.ndarray:
    """Compute yaw angle from direction vectors.

    Vectorized version of MATLAB ``find_yawangle.m``: zeros the Z component,
    normalizes, then computes ``arccos(dot(v, [0, 1, 0]))``.

    Parameters
    ----------
    vectors : np.ndarray
        (N, 3) array of direction vectors (e.g. head-to-tail after pitch
        correction).

    Returns
    -------
    np.ndarray
        (N,) yaw angles in degrees (unsigned, 0-180).
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    v = vectors.copy()
    v[:, 2] = 0.0  # remove vertical (Z) component

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v = v / norms

    cos_theta = np.clip(v[:, 1], -1.0, 1.0)  # dot(v, [0, 1, 0])
    return np.degrees(np.arccos(cos_theta))


def build_rotation_matrices(
    angles_deg: np.ndarray,
    axis: str = "x",
) -> np.ndarray:
    """Build (N, 3, 3) rotation matrices from angle arrays.

    Parameters
    ----------
    angles_deg : np.ndarray
        (N,) array of rotation angles in degrees.
    axis : str
        Rotation axis: ``"x"``, ``"y"``, or ``"z"``.

    Returns
    -------
    np.ndarray
        (N, 3, 3) array of rotation matrices.
    """
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    n = len(angles)
    c = np.cos(angles)
    s = np.sin(angles)
    zeros = np.zeros(n)
    ones = np.ones(n)

    if axis == "x":
        # Rx(theta)
        R = np.array([
            [ones, zeros, zeros],
            [zeros, c, -s],
            [zeros, s, c],
        ])  # shape (3, 3, N)
    elif axis == "y":
        R = np.array([
            [c, zeros, s],
            [zeros, ones, zeros],
            [-s, zeros, c],
        ])
    elif axis == "z":
        # MATLAB uses rotz(-angle), so negate the angle
        R = np.array([
            [c, s, zeros],
            [-s, c, zeros],
            [zeros, zeros, ones],
        ])
    else:
        msg = f"axis must be 'x', 'y', or 'z', got {axis!r}"
        raise ValueError(msg)

    # Transpose from (3, 3, N) to (N, 3, 3)
    return R.transpose(2, 0, 1)


def apply_rotation(
    xyz: np.ndarray,
    rotation_matrices: np.ndarray,
) -> np.ndarray:
    """Apply per-point rotation matrices to coordinate vectors.

    Equivalent to MATLAB ``xyz * rotmat`` (row vectors), implemented as
    ``R @ v`` with column convention via einsum.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) array of coordinates.
    rotation_matrices : np.ndarray
        (N, 3, 3) rotation matrices.

    Returns
    -------
    np.ndarray
        (N, 3) rotated coordinates.
    """
    # MATLAB uses row-vector convention: xyz * R
    # In NumPy column convention: R^T @ v, or equivalently v @ R
    return np.einsum("ni,nij->nj", xyz, rotation_matrices)


def build_body_frame(
    tail_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a body-fixed coordinate frame from tail vectors.

    Reproduces MATLAB lines 2253-2274 of ``run_whole_body_analysis.m``:

    1. ``body_axis`` = normalized tail_vector (backpack -> tailpack direction)
    2. ``sideways`` = cross([0,0,1], body_axis), normalized
    3. ``upwards`` = cross(body_axis, sideways), normalized
    4. ``sideways`` = cross(body_axis, upwards), normalized (refinement)

    Parameters
    ----------
    tail_vectors : np.ndarray
        (N, 3) vectors from backpack to tailpack.

    Returns
    -------
    body_axis : np.ndarray
        (N, 3) unit vectors along the body axis.
    sideways : np.ndarray
        (N, 3) unit vectors pointing sideways (left).
    upwards : np.ndarray
        (N, 3) unit vectors pointing up from the body.
    """
    tail_vectors = np.asarray(tail_vectors, dtype=float)

    # body_axis = normalized tail vector
    norms = np.linalg.norm(tail_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    body_axis = tail_vectors / norms

    # sideways = cross([0,0,1], body_axis), normalized
    z_up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), body_axis.shape)
    sideways = np.cross(z_up, body_axis)
    sw_norms = np.linalg.norm(sideways, axis=1, keepdims=True)
    sw_norms = np.where(sw_norms == 0, 1.0, sw_norms)
    sideways = sideways / sw_norms

    # upwards = cross(body_axis, sideways), normalized
    upwards = np.cross(body_axis, sideways)
    up_norms = np.linalg.norm(upwards, axis=1, keepdims=True)
    up_norms = np.where(up_norms == 0, 1.0, up_norms)
    upwards = upwards / up_norms

    # Refine sideways = cross(body_axis, upwards), normalized
    sideways = np.cross(body_axis, upwards)
    sw_norms = np.linalg.norm(sideways, axis=1, keepdims=True)
    sw_norms = np.where(sw_norms == 0, 1.0, sw_norms)
    sideways = sideways / sw_norms

    return body_axis, sideways, upwards


def rotate_to_body_frame(
    xyz: np.ndarray,
    sideways: np.ndarray,
    body_axis: np.ndarray,
    upwards: np.ndarray,
) -> np.ndarray:
    """Rotate marker positions into body-fixed coordinates.

    Constructs rotation matrices from the three body-frame axes and applies
    them via einsum. Reproduces MATLAB's ``global2localcoord`` with
    ``splitapply(@rotate_byRow, ...)``.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) marker positions in the global frame (relative to backpack).
    sideways : np.ndarray
        (N, 3) sideways unit vectors (body-frame X axis).
    body_axis : np.ndarray
        (N, 3) body-axis unit vectors (body-frame Y axis).
    upwards : np.ndarray
        (N, 3) upwards unit vectors (body-frame Z axis).

    Returns
    -------
    np.ndarray
        (N, 3) coordinates in the body-fixed frame.
    """
    # Build rotation matrix: rows are the body-frame axes
    # R[n] = [[sideways], [body_axis], [upwards]]
    R = np.stack([sideways, body_axis, upwards], axis=1)  # (N, 3, 3)
    return np.einsum("nij,nj->ni", R, xyz)


def extract_body_angles(
    body_axis: np.ndarray,
    sideways: np.ndarray,
    upwards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract pitch, yaw, and roll angles from body-frame axes.

    Reproduces the angle extraction in MATLAB ``rotate_byRow.m``:
    - Pitch: from body_axis using ``find_pitchangle`` logic
    - Yaw: from body_axis projected onto XY plane
    - Roll: from body_axis projected onto XZ plane

    Also applies the MATLAB wrap-around corrections (lines 2294-2303).

    Parameters
    ----------
    body_axis : np.ndarray
        (N, 3) body-axis unit vectors.
    sideways : np.ndarray
        (N, 3) sideways unit vectors.
    upwards : np.ndarray
        (N, 3) upwards unit vectors.

    Returns
    -------
    pitch : np.ndarray
        (N,) body pitch in degrees.
    yaw : np.ndarray
        (N,) body yaw in degrees.
    roll : np.ndarray
        (N,) body roll in degrees.
    """
    # Pitch from body_axis (= y_local_axis in MATLAB)
    pitch = compute_pitch_angle(body_axis)

    # Fix pitch wrapping (MATLAB lines 2294-2297)
    pitch = np.where(pitch < -90, pitch + 360, pitch)
    pitch = np.where(pitch > 120, pitch - 145, pitch)

    # Yaw from body_axis projected onto XY plane
    v_yaw = body_axis.copy()
    v_yaw[:, 2] = 0.0
    yaw_norms = np.linalg.norm(v_yaw, axis=1, keepdims=True)
    yaw_norms = np.where(yaw_norms == 0, 1.0, yaw_norms)
    v_yaw = v_yaw / yaw_norms

    yaw = np.degrees(np.arccos(np.clip(v_yaw[:, 1], -1.0, 1.0)))

    # Sign from cross product with [0,1,0] dotted with upwards
    cross_yaw = np.cross(v_yaw, np.array([0.0, 1.0, 0.0]))
    sign = np.sign(np.einsum("ij,ij->i", cross_yaw, upwards))
    yaw = np.where(sign > 0, yaw, -yaw)

    # Fix yaw wrapping (MATLAB lines 2300-2303)
    yaw = np.where(yaw > 90, yaw - 180, yaw)
    yaw = np.where(yaw < -90, yaw + 180, yaw)

    # Euler angles via rotation matrix for roll
    R = np.stack([sideways, body_axis, upwards], axis=1)  # (N, 3, 3)
    # ZYX Euler: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # roll = atan2(R[2,1], R[2,2]) = atan2(R[:,2,1], R[:,2,2])
    roll = np.degrees(np.arctan2(R[:, 2, 0], R[:, 2, 2]))

    return pitch, yaw, roll
