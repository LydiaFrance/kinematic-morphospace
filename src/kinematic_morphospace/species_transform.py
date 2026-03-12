"""Piecewise marker-by-marker transformation for cross-species comparison."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from morphing_birds import Animal3D
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

from .cross_species import integrate_dataframe_to_bird3D

logger = logging.getLogger(__name__)

def compute_transformation_matrix(source_marker: np.ndarray, target_marker: np.ndarray) -> np.ndarray:
    """
    Computes a 3x3 transformation matrix that rotates and scales
    source_marker to align with target_marker.
    
    Args:
        source_marker: Source marker position (3D vector)
        target_marker: Target marker position (3D vector)
        
    Returns:
        3x3 transformation matrix combining rotation and scaling
    """
    # Compute norms (magnitudes)
    norm_source = np.linalg.norm(source_marker)
    norm_target = np.linalg.norm(target_marker)
    if norm_source == 0:
        raise ValueError("Source marker has zero length.")
    if norm_target == 0:
        raise ValueError("Target marker has zero length.")

    # Scaling factor: ratio of lengths
    scale = norm_target / norm_source

    # Compute rotation axis (cross product) and angle (via dot product)
    axis = np.cross(source_marker, target_marker)
    axis_norm = np.linalg.norm(axis)
    if axis_norm != 0:
        axis = axis / axis_norm
    else:
        # Vectors are parallel or anti-parallel: pick an arbitrary perpendicular axis
        # Use whichever canonical axis is least aligned with source_marker
        abs_src = np.abs(source_marker / norm_source)
        min_idx = np.argmin(abs_src)
        perp = np.zeros(3)
        perp[min_idx] = 1.0
        axis = np.cross(source_marker, perp)
        axis = axis / np.linalg.norm(axis)
    
    dot_product = np.dot(source_marker, target_marker)
    cos_angle = np.clip(dot_product / (norm_source * norm_target), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Create rotation matrix from axis-angle representation
    rotation = R.from_rotvec(angle * axis)
    rotation_matrix = rotation.as_matrix()

    # Combine rotation and scaling
    T = scale * rotation_matrix
    return T

def transform_hawk_to_species(hawk_3d: Any,
                            species_idx: int,
                            species_df: pd.DataFrame,
                            tail_z_override: float = -0.05) -> Tuple[Animal3D, Animal3D, np.ndarray]:
    """
    Transform a hawk shape to match a target species using marker-by-marker transformation.

    Args:
        hawk_3d: Animal3D object containing hawk shape
        species_idx: Index of the target species in the dataset
        species_df: DataFrame containing species data
        tail_z_override: Z-coordinate override for the tail-tip marker after
            transformation.  Cadaver-derived tail positions tend to droop
            relative to live birds; this override corrects the tail-tip z
            to a biologically plausible value.  Set to ``None`` to skip
            the override entirely.

    Returns:
        Tuple containing:
        - transformed_bird_3d: Transformed bird shape
        - target_bird_3d: Target bird shape
        - transformation_matrix: Block diagonal matrix containing all transformations
    """
    # Get target bird markers
    markers_dict = integrate_dataframe_to_bird3D(
        species_df,
        row_idx=species_idx
    )
    logger.info("Species selected: %s", species_df['species_common'].iloc[species_idx])

    # Create target bird instance
    target_bird_3d = Animal3D('hawk', data=markers_dict)
    
    # Get marker positions
    hawk_markers = hawk_3d.right_markers.reshape(-1, 3)
    target_markers = target_bird_3d.right_markers.reshape(-1, 3)
    
    # Compute transformation matrix for each marker
    T_list = [compute_transformation_matrix(hawk_markers[i], target_markers[i]) 
              for i in range(len(hawk_markers))]
    
    # Build block-diagonal transformation matrix
    T = block_diag(*T_list)
    
    # Apply transformation to hawk markers
    transformed_markers = (T @ hawk_markers.reshape(-1)).reshape(-1, 3)
    
    # Adjust tail tip z-coordinate to correct for cadaver tail droop
    if tail_z_override is not None:
        transformed_markers[-1, 2] = tail_z_override
    
    # Reshape and create bilateral markers
    transformed_markers = transformed_markers.reshape(1, np.shape(transformed_markers)[0], 3)
    bilateral_markers = hawk_3d.mirror_keypoints(transformed_markers)
    
    # Create marker dictionary (moving markers from transformation)
    transformed_marker_dict = create_marker_dict(
        bilateral_markers.reshape(-1, 3),
        target_bird_3d.skeleton_definition.analysis_markers
    )

    # Merge fixed markers from the target species into the transformed dict
    fixed_marker_names = {"shoulder", "tailbase", "hood"}
    for key, val in markers_dict.items():
        base = key.split("_", 1)[-1] if "_" in key else key
        if base in fixed_marker_names or key in fixed_marker_names:
            transformed_marker_dict[key] = val

    # Create transformed bird instance
    transformed_bird_3d = Animal3D('hawk', data=transformed_marker_dict)
    
    return transformed_bird_3d, target_bird_3d, T

def transform_principal_components(principal_components: np.ndarray, 
                                transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Transform principal components using the block diagonal transformation matrix.
    
    Args:
        principal_components: Array containing principal components
        transformation_matrix: Block diagonal matrix containing all transformations
        
    Returns:
        Transformed principal components
    """
    pc_shape = principal_components.shape
    pc_reshaped = principal_components.reshape(pc_shape[0], -1)
    pc_transformed = transformation_matrix @ pc_reshaped.T
    pc_transformed = pc_transformed.T.reshape(pc_shape)
    return pc_transformed

def create_marker_dict(bilateral_markers: np.ndarray, marker_names: list) -> Dict[str, list]:
    """Create a dictionary mapping marker names to their 3-D coordinates.

    Parameters
    ----------
    bilateral_markers : np.ndarray
        Array of shape ``(n_markers, 3)`` containing marker coordinates.
    marker_names : list of str
        Marker names corresponding to the rows of *bilateral_markers*.

    Returns
    -------
    dict
        Dictionary mapping each marker name to an ``[x, y, z]`` list.
    """
    marker_dict = {}
    for i, name in enumerate(marker_names):
        marker_dict[name] = bilateral_markers[i].tolist()
    return marker_dict