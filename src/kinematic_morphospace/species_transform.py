"""Piecewise marker-by-marker transformation for cross-species comparison."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from morphing_birds import Animal3D
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

from .cross_species import integrate_dataframe_to_bird3D

logger = logging.getLogger(__name__)


def compute_transformation_matrix(
    source_marker: np.ndarray, target_marker: np.ndarray
) -> np.ndarray:
    """Compute a 3x3 rotation and scale matrix for source to target marker.

    Combines a rotation (aligning the source vector direction with the target)
    and a uniform scale (matching their magnitudes) into a single 3x3 matrix.
    This is used as the per-marker building block of the block-diagonal
    species transformation.

    Args:
        source_marker: Source marker position as a 3-D vector.
        target_marker: Target marker position as a 3-D vector.

    Returns:
        3x3 transformation matrix combining rotation and scaling.

    Raises:
        ValueError: If either marker has zero length.
    """
    norm_source = np.linalg.norm(source_marker)
    norm_target = np.linalg.norm(target_marker)
    if norm_source == 0:
        msg = "Source marker has zero length."
        raise ValueError(msg)
    if norm_target == 0:
        msg = "Target marker has zero length."
        raise ValueError(msg)

    scale = norm_target / norm_source

    axis = np.cross(source_marker, target_marker)
    axis_norm = np.linalg.norm(axis)
    if axis_norm != 0:
        axis = axis / axis_norm
    else:
        # Vectors are parallel or anti-parallel: pick an arbitrary perpendicular axis
        abs_src = np.abs(source_marker / norm_source)
        min_idx = np.argmin(abs_src)
        perp = np.zeros(3)
        perp[min_idx] = 1.0
        axis = np.cross(source_marker, perp)
        axis = axis / np.linalg.norm(axis)

    dot_product = np.dot(source_marker, target_marker)
    cos_angle = np.clip(dot_product / (norm_source * norm_target), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    rotation = R.from_rotvec(angle * axis)
    rotation_matrix = rotation.as_matrix()

    return scale * rotation_matrix


def _get_body_axis(markers_dict: dict) -> np.ndarray:
    """Return the shoulder-midpoint → tailbase-midpoint vector from a marker dict."""
    ls = np.array(markers_dict["left_shoulder"])
    rs = np.array(markers_dict["right_shoulder"])
    lt = np.array(markers_dict["left_tailbase"])
    rt = np.array(markers_dict["right_tailbase"])
    shoulder_mid = (ls + rs) / 2
    tailbase_mid = (lt + rt) / 2
    return tailbase_mid - shoulder_mid


def align_body_axis(
    markers_dict: dict,
    hawk_3d: Any,
) -> dict:
    """Rigidly rotate species markers so body pitch matches the hawk reference.

    The body axis is defined as shoulder-midpoint → tailbase-midpoint.
    Both vectors lie in the y-z plane (x ≈ 0 by bilateral symmetry),
    so a single pitch rotation around the x-axis is sufficient.
    Rotation is applied about the shoulder midpoint.

    Args:
        markers_dict: Marker name → [x, y, z] dictionary from
            :func:`~.cross_species.integrate_dataframe_to_bird3D`.
        hawk_3d: Animal3D object containing the hawk reference shape.

    Returns:
        New marker dictionary with all positions pitch-corrected.
    """
    target_body = _get_body_axis(markers_dict)
    target_angle = np.arctan2(target_body[2], target_body[1])

    all_markers = np.concatenate(
        [hawk_3d.markers.reshape(-1, 3), hawk_3d.fixed_markers.reshape(-1, 3)]
    )
    names = hawk_3d.skeleton.all_marker_names
    hawk_smid = (all_markers[names.index("left_shoulder")]
                 + all_markers[names.index("right_shoulder")]) / 2
    hawk_tmid = (all_markers[names.index("left_tailbase")]
                 + all_markers[names.index("right_tailbase")]) / 2
    hawk_body = hawk_tmid - hawk_smid
    hawk_angle = np.arctan2(hawk_body[2], hawk_body[1])

    theta = hawk_angle - target_angle
    if abs(theta) < 1e-8:
        return dict(markers_dict)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t],
    ])

    ls = np.array(markers_dict["left_shoulder"])
    rs = np.array(markers_dict["right_shoulder"])
    pivot = (ls + rs) / 2

    rotated = {}
    for name, coords in markers_dict.items():
        v = np.array(coords) - pivot
        rotated[name] = (rot_x @ v + pivot).tolist()

    # Translate so shoulder midpoint z matches the hawk
    rotated_smid = (np.array(rotated["left_shoulder"])
                    + np.array(rotated["right_shoulder"])) / 2
    z_offset = hawk_smid[2] - rotated_smid[2]
    for name in rotated:
        rotated[name][2] += z_offset

    return rotated


def transform_hawk_to_species(
    hawk_3d: Any,
    species_idx: int,
    species_df: pd.DataFrame,
    tail_z_override: float | None = -0.05,
) -> tuple[Animal3D, Animal3D, np.ndarray]:
    """Transform hawk shape to target species via marker-by-marker transformation.

    Args:
        hawk_3d: Animal3D object containing the hawk reference shape.
        species_idx: Index of the target species row in ``species_df``.
        species_df: DataFrame containing species morphological data from
            the Harvey et al. dataset.
        tail_z_override: Z-coordinate override for the tail-tip marker after
            transformation. Cadaver-derived tail positions tend to droop
            relative to live birds; set to None to skip the override.
            Defaults to -0.05.

    Returns:
        Tuple containing the transformed bird shape (Animal3D), the target
        bird shape (Animal3D), and the block-diagonal transformation matrix
        of shape ``(3*n_markers, 3*n_markers)``.
    """
    markers_dict = integrate_dataframe_to_bird3D(
        species_df,
        row_idx=species_idx
    )
    logger.info("Species selected: %s", species_df['species_common'].iloc[species_idx])

    markers_dict = align_body_axis(markers_dict, hawk_3d)

    target_bird_3d = Animal3D('hawk', data=markers_dict)

    hawk_markers = hawk_3d.right_markers.reshape(-1, 3)
    target_markers = target_bird_3d.right_markers.reshape(-1, 3)

    T_list = [compute_transformation_matrix(hawk_markers[i], target_markers[i])
              for i in range(len(hawk_markers))]

    T = block_diag(*T_list)

    transformed_markers = (T @ hawk_markers.reshape(-1)).reshape(-1, 3)

    if tail_z_override is not None:
        transformed_markers[-1, 2] = tail_z_override

    n_markers = np.shape(transformed_markers)[0]
    transformed_markers = transformed_markers.reshape(1, n_markers, 3)
    bilateral_markers = hawk_3d.mirror_keypoints(transformed_markers)

    transformed_marker_dict = create_marker_dict(
        bilateral_markers.reshape(-1, 3),
        target_bird_3d.skeleton.analysis_markers
    )

    fixed_marker_names = {"shoulder", "tailbase", "hood"}
    for key, val in markers_dict.items():
        base = key.split("_", 1)[-1] if "_" in key else key
        if base in fixed_marker_names or key in fixed_marker_names:
            transformed_marker_dict[key] = val

    transformed_bird_3d = Animal3D('hawk', data=transformed_marker_dict)

    return transformed_bird_3d, target_bird_3d, np.asarray(T)


def transform_principal_components(
    principal_components: np.ndarray,
    transformation_matrix: np.ndarray,
) -> np.ndarray:
    """Transform principal components using block-diagonal transformation matrix.

    Applies the same marker-by-marker transformation used for the mean shape
    to each principal component vector, mapping the hawk morphospace axes
    into the target species' coordinate frame.

    Args:
        principal_components: Array of shape ``(n_components, n_features)``
            containing the PCA components to transform.
        transformation_matrix: Block-diagonal transformation matrix of shape
            ``(3*n_markers, 3*n_markers)`` from :func:`transform_hawk_to_species`.

    Returns:
        Transformed principal component array of the same shape as
        ``principal_components``.
    """
    pc_shape = principal_components.shape
    pc_reshaped = principal_components.reshape(pc_shape[0], -1)
    pc_transformed = transformation_matrix @ pc_reshaped.T
    return pc_transformed.T.reshape(pc_shape)


def create_marker_dict(
    bilateral_markers: np.ndarray, marker_names: list
) -> dict[str, list]:
    """Build a marker name → coordinate dictionary from an array of 3-D positions.

    Args:
        bilateral_markers: Array of shape ``(n_markers, 3)`` containing
            marker coordinates.
        marker_names: Marker names corresponding to the rows of
            ``bilateral_markers``.

    Returns:
        Dictionary mapping each marker name to an ``[x, y, z]`` list.
    """
    marker_dict = {}
    for i, name in enumerate(marker_names):
        marker_dict[name] = bilateral_markers[i].tolist()
    return marker_dict
