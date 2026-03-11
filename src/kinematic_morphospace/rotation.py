import numpy as np
from scipy.spatial.transform import Rotation as R


# --------- Symmetry and Kabsch rotation ---------


def assess_symmetry(pc, axis='x', nMarkers=8):
    """
    Calculate symmetry score for a given principal component.

    Returns the **sum of squared differences** between each left marker
    and its mirrored right marker.  This is *not* RMS — to convert,
    use ``np.sqrt(score / nMarkers)``.

    Allows mirroring around 'x', 'y', or 'z' axes to check for symmetry.
    Assumes marker pairs are structured as (0,1), (2,3), (4,5), (6,7).
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
    """Vectorised Kabsch algorithm that computes the optimal rotation matrices.

    Parameters
    ----------
    P, Q : ndarray, shape (n, markers, 3)
        Point clouds to align. P is rotated onto Q.
    centre : bool, default False
        If True, subtract per-frame centroid before computing the rotation.
        The original pipeline did not centre (the markers are already
        backpack-relative), so the default is False for reproducibility.
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
    """Extract the Euler angles (roll, pitch, yaw) from the rotation matrices."""
    rotations = R.from_matrix(rotation_matrices)
    euler_angles = rotations.as_euler(sequence, degrees=True)
    return euler_angles


def apply_rotation(P, rotation_matrices):
    """Apply rotation matrices to the original data."""
    P_transformed = np.empty_like(P)

    for i in range(P_transformed.shape[0]):
        P_transformed[i] = np.dot(P[i], rotation_matrices[i])

    return P_transformed


# --------- Body rotation corrections ---------


def undo_body_pitch_rotation(markers, body_pitch):
    """
    Undo the body pitch rotation of the markers.

    Parameters:
    - markers (numpy.ndarray): A numpy array containing the markers.
    - body_pitch (numpy.ndarray): A numpy array containing the body pitch angles.

    Returns:
    - corrected_markers (numpy.ndarray): Markers with body pitch rotation removed.
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
    """
    Undo the body rotation of the markers around a specified axis.

    ``which_axis`` selects which plane the measured body angle lies in,
    **not** the geometric rotation axis.  The mapping is:

    * ``'z'`` — angle measured in the yz-plane → applies **Rx** (x-axis
      rotation).  Identical to :func:`undo_body_pitch_rotation`.
    * ``'x'`` — angle measured in the xz-plane → applies **Ry** (y-axis
      rotation).
    * ``'y'`` — angle measured in the xy-plane → applies **Rz** (z-axis
      rotation).

    Parameters:
    - markers (numpy.ndarray): A numpy array containing the markers.
    - whole_body_angle (numpy.ndarray): A numpy array containing the body angles
      (degrees).
    - which_axis (str): The correction plane ('x', 'y', or 'z'). See above for
      the mapping to geometric rotation axes.

    Returns:
    - corrected_markers (numpy.ndarray): Markers with body rotation removed.
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
