"""Core PCA routines for bird wing morphospace analysis.

Provides functions to run PCA on marker data, project onto fitted
components, and run per-bird PCA for individual-level comparisons.
"""

from sklearn.decomposition import PCA

from .data_filtering import filter_by

# ------- PCA -------

def run_PCA(markers, project_data=None, n_components=None, flat_input=False):
    """Run Principal Component Analysis on marker data.

    Fits a PCA model on ``markers`` and projects the data (or optional
    ``project_data``) onto the resulting components. This is the primary
    entry point for building the morphospace.

    Args:
        markers: Input marker data of shape ``(N, n_markers, 3)`` or
            ``(N, n_features)`` if ``flat_input=True``.
        project_data: Optional array to project onto the fitted PCA space.
            If None, ``markers`` itself is projected. Defaults to None.
        n_components: Number of components to retain. If None, all components
            are kept (existing behaviour). Defaults to None.
        flat_input: If True, skip reshaping — input is already
            ``(N, n_features)``. Defaults to False.

    Returns:
        Tuple of (principal_components, scores, pca) where
        principal_components has shape ``(n_components, n_features)``,
        scores has shape ``(N, n_components)``, and pca is the fitted
        sklearn PCA object.

    Raises:
        ValueError: If the input data shapes are inconsistent or the output
            shapes do not match expectations.
    """
    pca_input = markers if flat_input else get_PCA_input(markers)

    pca = PCA(n_components=n_components, random_state=0)
    pca_output = pca.fit(pca_input)

    if project_data is None:
        project_data = pca_input
    elif flat_input:
        pass  # project_data is already flat
    else:
        project_data = get_PCA_input(project_data)

    principal_components = pca_output.components_
    scores = pca_output.transform(project_data)

    # Validate output shapes
    if n_components is None and not flat_input:
        try:
            test_PCA_output(project_data, principal_components, scores)
        except AssertionError as err:
            msg = f"PCA output validation failed: {err!s}"
            raise ValueError(msg) from err
    else:
        n_features = pca_input.shape[1]
        n_out = n_components if n_components is not None else n_features
        if principal_components.shape != (n_out, n_features):
            msg = (
                f"Components shape {principal_components.shape} does not match "
                f"expected ({n_out}, {n_features})"
            )
            raise ValueError(
                msg)
        if scores.shape[1] != n_out:
            msg = f"Scores have {scores.shape[1]} columns, expected {n_out}"
            raise ValueError(
                msg)

    return principal_components, scores, pca


def run_PCA_birds(markers, frame_info_df, filter_on=True, birds=None, year=None):
    """Run per-bird PCA and return principal components keyed by bird name.

    Fits a separate PCA model for each bird using straight-flight frames
    (obstacle=0), enabling comparison of individual morphospaces. By default
    uses each bird's most recent recording year (Period 2 / 2020 where
    available) to compare experienced adults.

    Args:
        markers: Marker array of shape ``(n_frames, n_markers, 3)``.
        frame_info_df: DataFrame of per-frame metadata containing ``BirdID``
            and ``Year`` columns.
        filter_on: If True (default), restrict each bird to straight-flight
            frames (``obstacle=0``).
        birds: List of bird name strings to process. If None, all birds
            present in ``frame_info_df`` are included. Defaults to None.
        year: Recording year to use for all birds. If None (default), the
            most recent available year is used per bird.

    Returns:
        Dict mapping bird name (str) to principal component array of shape
        ``(n_features, n_features)``.
    """
    hawk_id_to_name = {1: "Drogon", 2: "Rhaegal", 3: "Ruby",
                       4: "Toothless", 5: "Charmander"}

    if birds is None:
        birds = [
            hawk_id_to_name[bid]
            for bid in sorted(frame_info_df["BirdID"].unique())
            if bid in hawk_id_to_name
        ]

    components_by_bird = {}
    for bird in birds:
        if year is not None:
            bird_year = year
        else:
            bird_mask = filter_by(frame_info_df, hawkname=bird)
            bird_year = int(frame_info_df["Year"][bird_mask].max())

        if filter_on:
            filt = filter_by(frame_info_df, hawkname=bird, obstacle=0, year=bird_year)
        else:
            filt = filter_by(frame_info_df, hawkname=bird, year=bird_year)
        components_by_bird[bird], _, _ = run_PCA(markers[filt])

    return components_by_bird


# ....... Helper functions .......

def get_PCA_input_sizes(pca_input):
    """Extract frame count, marker count, and feature count from a flat PCA input.

    Args:
        pca_input: 2-D array of shape ``(n_frames, n_features)`` where
            ``n_features = n_markers * 3``.

    Returns:
        Tuple of (n_frames, n_markers, n_vars).
    """
    n_frames = pca_input.shape[0]
    n_markers = pca_input.shape[1] / 3
    n_vars = pca_input.shape[1]

    return n_frames, n_markers, n_vars


def get_PCA_input(markers):
    """Reshape a 3-D marker array to a 2-D PCA input matrix.

    Args:
        markers: Marker array of shape ``(n_frames, n_markers, 3)``.

    Returns:
        Array of shape ``(n_frames, n_markers * 3)`` suitable for sklearn PCA.
    """
    n_markers = markers.shape[1]
    return markers.reshape(-1, n_markers * 3)



def test_PCA_output(pca_input, principal_components, scores):
    """Assert that PCA output shapes are self-consistent.

    Args:
        pca_input: 2-D array of shape ``(n_frames, n_vars)``.
        principal_components: Components array of shape ``(n_vars, n_vars)``.
        scores: Score array of shape ``(n_frames, n_vars)``.

    Raises:
        AssertionError: If any shape invariant is violated.
    """
    n_frames, n_markers, n_vars = get_PCA_input_sizes(pca_input)

    assert n_vars == n_markers * 3, "n_vars is not equal to n_markers*3."
    assert principal_components.shape[0] == n_vars, (
        "principal_components is not the right shape."
    )
    assert principal_components.shape[1] == n_vars, (
        "principal_components is not the right shape."
    )
    assert scores.shape[0] == n_frames, (
        "scores first dim is not the right shape."
    )
    assert scores.shape[1] == n_vars, (
        "scores second dim is not the right shape."
    )
