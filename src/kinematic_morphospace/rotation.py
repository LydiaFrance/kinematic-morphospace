"""Rotation utilities for symmetry assessment and body-pitch correction.

Provides functions for computing the Kabsch optimal rotation, extracting
Euler angles, assessing bilateral symmetry of principal components, and
removing whole-body pitch and roll from marker data prior to PCA.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# --------- Symmetry and Kabsch rotation ---------


def assess_symmetry(pc, axis='x', nMarkers=8):
    """Calculate the sum of squared left-right differences for a principal component.

    Returns the **sum of squared differences** between each left marker
    and its mirrored right marker.  This is *not* RMS — to convert,
    use ``np.sqrt(score / nMarkers)``.

    Assumes marker pairs are structured as (0, 1), (2, 3), (4, 5), (6, 7)
    (left/right interleaved). A score of zero indicates perfect bilateral
    symmetry; higher values indicate greater asymmetry.

    Args:
        pc: Principal component vector reshaped to ``(n_bilateral_markers, 3)``.
        axis: Axis to mirror for symmetry comparison. Use ``'x'`` for the
            standard left/right midline. Defaults to ``'x'``.
        nMarkers: Total number of bilateral markers (left + right). Defaults to 8.

    Returns:
        Sum of squared differences between mirrored marker pairs.

    Raises:
        ValueError: If ``axis`` is not ``'x'``, ``'y'``, or ``'z'``.
    """
    axis_to_index = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_to_index:
        raise ValueError(f"Invalid axis '{axis}'. Use 'x', 'y', or 'z'.")

    symmetry = 0
    for ii in range(0, nMarkers, 2):
        left_marker = pc[ii]
        right_marker = pc[ii + 1]

        mirrored_right_marker = np.copy(right_marker)
        mirrored_right_marker[axis_to_index[axis]] *= -1

        symmetry += np.sum((left_marker - mirrored_right_marker) ** 2)

    return symmetry


def vectorised_kabsch(P, Q, centre=False):
    """Compute the optimal rotation matrices aligning P to Q for each frame.

    Uses the vectorised Kabsch algorithm (SVD-based) to find the rotation
    matrix that minimises the RMSD between ``P`` and ``Q`` for every frame
    simultaneously. Used to remove whole-body rotational drift from flight
    sequences.

    Args:
        P: Point cloud array of shape ``(n, markers, 3)`` to be rotated.
        Q: Reference point cloud array of shape ``(n, markers, 3)``.
        centre: If True, subtract the per-frame centroid before computing
            the rotation. The original pipeline does not centre (markers are
            already backpack-relative), so the default is False for
            reproducibility. Defaults to False.

    Returns:
        Rotation matrix array of shape ``(n, 3, 3)``.

    Raises:
        ValueError: If P and Q have mismatched shapes.
    """
    if P.shape[0] != Q.shape[0]:
        raise ValueError(
            f"P and Q must have the same batch size (first dimension). "
            f"Got P.shape={P.shape} and Q.shape={Q.shape}. "
            f"P has {P.shape[0]} instances but Q has {Q.shape[0]} instances."
        )
    if P.shape[1:] != Q.shape[1:]:
        raise ValueError(
            f"P and Q must have the same shape for markers and coordinates. "
            f"Got P.shape={P.shape} and Q.shape={Q.shape}."
        )

    if centre:
        P_centred = P - P.mean(axis=1, keepdims=True)
        Q_centred = Q - Q.mean(axis=1, keepdims=True)
    else:
        P_centred = P
        Q_centred = Q

    # Compute the cross-covariance matrix for each instance
    C = np.einsum('nik,nij->nkj', P_centred, Q_centred)

    # Perform SVD for each cross-covariance matrix
    U, S, Vt = np.linalg.svd(C)

    # Compute the optimal rotation matrices
    d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0
    Vt[d, -1, :] *= -1
    rotation_matrices = np.einsum('nij,njk->nik', U, Vt)

    return rotation_matrices


def extract_euler_angles_from_matrices(rotation_matrices, sequence='xyz'):
    """Extract Euler angles (roll, pitch, yaw) from a batch of rotation matrices.

    Args:
        rotation_matrices: Array of shape ``(n, 3, 3)`` containing rotation matrices.
        sequence: Euler angle convention string (e.g. ``'xyz'``, ``'zyx'``).
            Defaults to ``'xyz'``.

    Returns:
        Array of shape ``(n, 3)`` containing Euler angles in degrees.
    """
    rotations = R.from_matrix(rotation_matrices)
    euler_angles = rotations.as_euler(sequence, degrees=True)
    return euler_angles


def apply_rotation(P, rotation_matrices):
    """Apply per-frame rotation matrices to a marker array.

    Args:
        P: Marker array of shape ``(n_frames, n_markers, 3)``.
        rotation_matrices: Rotation matrix array of shape ``(n_frames, 3, 3)``.

    Returns:
        Rotated marker array of the same shape as ``P``.
    """
    P_transformed = np.empty_like(P)

    for i in range(P_transformed.shape[0]):
        P_transformed[i] = np.dot(P[i], rotation_matrices[i])

    return P_transformed


# --------- Body rotation corrections ---------


def undo_body_pitch_rotation(markers, body_pitch):
    """Remove the body pitch rotation from marker coordinates.

    Applies the inverse of the measured body pitch rotation (around the
    x-axis) to each frame, producing wing marker positions relative to a
    level body. This is required before PCA when body pitch varies
    systematically across the flight approach.

    Args:
        markers: Marker array of shape ``(n_frames, n_markers, 3)``.
        body_pitch: Array of body pitch angles in degrees, shape ``(n_frames,)``.

    Returns:
        Marker array of the same shape with body pitch rotation removed.
    """
    body_pitch_rad = np.radians(body_pitch)

    n_instances = markers.shape[0]
    corrected_markers = np.empty_like(markers)

    for i in range(n_instances):
        pitch = body_pitch_rad[i]

        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        corrected_markers[i] = markers[i] @ rotation_matrix.T

    return corrected_markers


def undo_body_rotation(markers, whole_body_angle, which_axis='z'):
    """Remove a whole-body rotation around a specified axis from marker coordinates.

    ``which_axis`` selects which plane the measured body angle lies in,
    **not** the geometric rotation axis.  The mapping is:

    * ``'z'`` — angle measured in the yz-plane → applies **Rx** (x-axis
      rotation).  Identical to :func:`undo_body_pitch_rotation`.
    * ``'x'`` — angle measured in the xz-plane → applies **Ry** (y-axis
      rotation).
    * ``'y'`` — angle measured in the xy-plane → applies **Rz** (z-axis
      rotation).

    Args:
        markers: Marker array of shape ``(n_frames, n_markers, 3)``.
        whole_body_angle: Array of body rotation angles in degrees, shape
            ``(n_frames,)``.
        which_axis: Correction plane (``'x'``, ``'y'``, or ``'z'``). See
            the note above for the mapping to geometric rotation axes.
            Defaults to ``'z'``.

    Returns:
        Marker array of the same shape with the body rotation removed.

    Raises:
        ValueError: If ``which_axis`` is not ``'x'``, ``'y'``, or ``'z'``.
    """
    body_pitch_rad = np.radians(whole_body_angle)

    n_instances = markers.shape[0]
    corrected_markers = np.empty_like(markers)

    for i in range(n_instances):
        pitch = body_pitch_rad[i]

        if which_axis == 'z':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ])
        elif which_axis == 'x':
            rotation_matrix = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
        elif which_axis == 'y':
            rotation_matrix = np.array([
                [np.cos(pitch), -np.sin(pitch), 0],
                [np.sin(pitch), np.cos(pitch), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"Invalid axis: {which_axis}. Use 'x', 'y', or 'z'.")

        corrected_markers[i] = markers[i] @ rotation_matrix.T

    return corrected_markers
