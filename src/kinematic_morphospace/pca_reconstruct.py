import numpy as np


def to_bilateral(right, left=None):
    """
    Assemble bilateral marker array from unilateral sides.

    If only *right* is provided, the left side is created by mirroring
    (negating the x-coordinate), producing a symmetric shape.

    If both *right* and *left* are provided, the left side is unmirrored
    (x-coordinate negated back) and interleaved with the right side,
    preserving any asymmetry.

    Parameters
    ----------
    right : np.ndarray, shape (n_frames, n_markers, 3)
        Right-side unilateral markers (as stored by PCA — x already positive).
    left : np.ndarray, optional, shape (n_frames, n_markers, 3)
        Left-side unilateral markers (x-mirrored convention, i.e. x positive).
        If None, the right side is mirrored to create a symmetric shape.

    Returns
    -------
    np.ndarray, shape (n_frames, 2 * n_markers, 3)
        Bilateral markers with left at even indices, right at odd indices.
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
    """
    Split bilateral marker array into stacked unilateral frames.

    Separates left and right markers, mirrors left x-coordinates so both
    sides share the same coordinate frame, and stacks them (left first,
    then right), doubling the frame count.

    Parameters
    ----------
    bilateral : np.ndarray, shape (n_frames, n_bilateral_markers, 3)
        Bilateral marker data with paired left/right markers.
    left_indices : list[int], optional
        Indices of left-side markers. Default: [0, 2, 4, 6].
    right_indices : list[int], optional
        Indices of right-side markers. Default: [1, 3, 5, 7].

    Returns
    -------
    np.ndarray, shape (2 * n_frames, n_markers_per_side, 3)
        Stacked unilateral data (left frames first, then right frames),
        with left x-coordinates mirrored.
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


def reconstruct(score_frames, principal_components, mu, components_list=None):
    """
    Reconstruct frames based on principal components and score frames.

    Parameters:
    - score_frames (numpy.ndarray): The score frames for reconstruction.
    - principal_components (numpy.ndarray): The principal components matrix.
    - mu (numpy.ndarray): The mean shape [1, n_markers, 3].
    - components_list (list): List of component indices to use. Defaults to all.

    Returns:
    - numpy.ndarray: The reconstructed frames.
    """

    if components_list is None:
        components_list = range(principal_components.shape[1])

    if not isinstance(score_frames, np.ndarray):
        raise TypeError("score_frames must be a numpy array.")

    if len(score_frames.shape) != 2:
        raise ValueError("score_frames must be 2d.")

    assert score_frames.shape[1] == principal_components.shape[0], \
        "score_frames must have the same number of columns as principal components."
    assert len(components_list) <= principal_components.shape[1], \
        "components_list must not exceed the number of principal components."
    assert len(mu.shape) == 3, "mu must be a 3d array: [1, nMarkers, 3]."

    n_markers = mu.shape[1]
    n_dims = mu.shape[2]
    n_frames = score_frames.shape[0]

    # Select principal components and scores based on the provided list
    selected_PCs = principal_components[components_list, :]
    selected_scores = score_frames[:, components_list]

    reconstruction = np.dot(selected_scores, selected_PCs)
    reconstruction = reconstruction.reshape(-1, n_markers, n_dims)

    reconstructed_frames = mu + reconstruction

    assert reconstructed_frames.shape[0] == n_frames
    assert reconstructed_frames.shape[1] == n_markers
    assert reconstructed_frames.shape[2] == n_dims

    return reconstructed_frames
