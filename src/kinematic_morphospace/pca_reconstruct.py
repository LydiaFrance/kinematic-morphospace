"""PCA reconstruction and bilateral/unilateral conversion utilities.

Provides functions to reconstruct marker positions from PCA scores and
to convert between bilateral (both wings) and unilateral (one wing,
mirrored) representations used throughout the morphospace pipeline.
"""

import numpy as np


def to_bilateral(right, left=None):
    """Assemble a bilateral marker array from unilateral sides.

    If only ``right`` is provided, the left side is created by mirroring
    (negating the x-coordinate), producing a symmetric shape.

    If both ``right`` and ``left`` are provided, the left side is unmirrored
    (x-coordinate negated back) and interleaved with the right side,
    preserving any asymmetry.

    Args:
        right: Right-side unilateral markers of shape ``(n_frames, n_markers, 3)``
            (x already positive, as stored by PCA).
        left: Left-side unilateral markers of shape ``(n_frames, n_markers, 3)``
            with x-mirrored convention (x positive). If None, the right side
            is mirrored to create a symmetric shape. Defaults to None.

    Returns:
        Bilateral marker array of shape ``(n_frames, 2 * n_markers, 3)`` with
        left markers at even indices and right markers at odd indices.
    """
    if left is None:
        left = right  # symmetric: both sides identical before mirroring

    n_frames, n_markers, n_dims = right.shape

    bilateral = np.empty((n_frames, 2 * n_markers, n_dims), dtype=right.dtype)

    # Left side: unmirror x-coordinate
    bilateral[:, ::2, :] = left
    bilateral[:, ::2, 0] *= -1

    # Right side: unchanged
    bilateral[:, 1::2, :] = right

    return bilateral


def to_unilateral(bilateral, left_indices=None, right_indices=None):
    """Split a bilateral marker array into stacked unilateral frames.

    Separates left and right markers, mirrors left x-coordinates so both
    sides share the same coordinate frame, and stacks them (left first,
    then right), doubling the frame count. This is the standard
    pre-processing step before PCA, as it exploits bilateral symmetry to
    double the effective sample size.

    Args:
        bilateral: Bilateral marker data of shape
            ``(n_frames, n_bilateral_markers, 3)``.
        left_indices: Indices of left-side markers. Defaults to [0, 2, 4, 6].
        right_indices: Indices of right-side markers. Defaults to [1, 3, 5, 7].

    Returns:
        Stacked unilateral array of shape ``(2 * n_frames, n_markers_per_side, 3)``
        with left frames first, then right, and left x-coordinates mirrored
        to match the right-side convention.
    """
    if left_indices is None:
        left_indices = [0, 2, 4, 6]
    if right_indices is None:
        right_indices = [1, 3, 5, 7]

    left = bilateral[:, left_indices, :]
    right = bilateral[:, right_indices, :]

    # Mirror left x-coordinates so both sides share the same frame
    left_mirrored = left.copy()
    left_mirrored[:, :, 0] *= -1

    return np.concatenate([left_mirrored, right], axis=0)


def prepare_unilateral(bilateral, frame_info_df, left_indices=None, right_indices=None):
    """Convert bilateral markers to unilateral and double the frame info.

    Combines :func:`to_unilateral` with the standard frame-info doubling
    (adding a 'Left' column to distinguish sides). This is the complete
    pre-processing step for bilateral-to-unilateral PCA.

    Args:
        bilateral: Bilateral marker data of shape
            ``(n_frames, n_bilateral_markers, 3)``.
        frame_info_df: Per-frame metadata DataFrame with one row per
            bilateral frame.
        left_indices: Indices of left-side markers. Defaults to [0, 2, 4, 6].
        right_indices: Indices of right-side markers. Defaults to [1, 3, 5, 7].

    Returns:
        Tuple of (unilateral_data, unilateral_frame_info) where
        unilateral_data has shape ``(2 * n_frames, n_markers_per_side, 3)``
        and unilateral_frame_info has 2 * n_frames rows with a 'Left' column.
    """
    import pandas as pd

    unilateral = to_unilateral(bilateral, left_indices, right_indices)

    left_info = frame_info_df.copy()
    left_info["Left"] = True
    right_info = frame_info_df.copy()
    right_info["Left"] = False
    unilateral_frame_info = pd.concat([left_info, right_info], ignore_index=True)

    return unilateral, unilateral_frame_info


def reconstruct(score_frames, principal_components, mu, components_list=None):
    """Reconstruct marker positions from PCA scores and principal components.

    Computes ``mu + selected_scores @ selected_PCs`` to recover the marker
    positions corresponding to given score values. Optionally restricts
    reconstruction to a subset of components, allowing visualisation of
    individual shape modes.

    Args:
        score_frames: Score array of shape ``(n_frames, n_components)``.
        principal_components: Principal component matrix of shape
            ``(n_components, n_features)``.
        mu: Mean shape array of shape ``(1, n_markers, 3)``.
        components_list: List of component indices to include in the
            reconstruction. Defaults to all components.

    Returns:
        Reconstructed marker array of shape ``(n_frames, n_markers, 3)``.

    Raises:
        TypeError: If ``score_frames`` is not a NumPy array.
        ValueError: If ``score_frames`` is not 2-D.
    """
    if components_list is None:
        components_list = range(principal_components.shape[1])

    if not isinstance(score_frames, np.ndarray):
        msg = "score_frames must be a numpy array."
        raise TypeError(msg)

    if len(score_frames.shape) != 2:
        msg = "score_frames must be 2d."
        raise ValueError(msg)

    assert score_frames.shape[1] == principal_components.shape[0], \
        "score_frames must have the same number of columns as principal components."
    assert len(components_list) <= principal_components.shape[1], \
        "components_list must not exceed the number of principal components."
    assert len(mu.shape) == 3, "mu must be a 3d array: [1, nMarkers, 3]."

    n_markers = mu.shape[1]
    n_dims = mu.shape[2]
    n_frames = score_frames.shape[0]

    selected_PCs = principal_components[components_list, :]
    selected_scores = score_frames[:, components_list]

    reconstruction = np.dot(selected_scores, selected_PCs)
    reconstruction = reconstruction.reshape(-1, n_markers, n_dims)

    reconstructed_frames = mu + reconstruction

    assert reconstructed_frames.shape[0] == n_frames
    assert reconstructed_frames.shape[1] == n_markers
    assert reconstructed_frames.shape[2] == n_dims

    return reconstructed_frames
