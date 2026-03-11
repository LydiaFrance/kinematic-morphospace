import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R


from .data_filtering import filter_by

# ------- PCA -------

def run_PCA(markers, project_data=None):
    """
    Run Principal Component Analysis on the given markers data.

    Args:
        markers (np.ndarray): Input marker data.
        project_data (np.ndarray, optional): Additional data to project onto the PCA space.

    Returns:
        Tuple[np.ndarray, np.ndarray, PCA]: Principal components, scores, and PCA object.

    Raises:
        ValueError: If the input data shapes are inconsistent.
    """
    # Reshape the data to be [n, nMarkers*3]
    pca_input = get_PCA_input(markers)

    # Run PCA
    pca = PCA(random_state=0)
    pca_output = pca.fit(pca_input)

    # User may want to fit the principal components
    # to a different dataset
    if project_data is None:
        project_data = pca_input
    else:
        project_data = get_PCA_input(project_data)

    # Another word for eigenvectors is components.
    principal_components = pca_output.components_
    
    # Another word for scores is projections.
    scores = pca_output.transform(project_data)

    # Check the shape of the output
    try:
        test_PCA_output(project_data, principal_components, scores)
    except AssertionError as msg:
        raise ValueError(f"PCA output validation failed: {str(msg)}")

    return principal_components, scores, pca

def run_PCA_birds(markers, frame_info_df, filter_on=True, birds=None, year=None):
    """Run per-bird PCA and return components keyed by bird name.

    Parameters
    ----------
    birds : list[str] or None
        Bird names to process.  When *None* (default), every unique bird
        in *frame_info_df* is included.
    year : int or None
        Year to filter each bird's data.  When *None* (default), the most
        recent year available for each bird is used (Period 2 / 2020 where
        available, so that experienced adults are compared).
    """
    hawk_id_to_name = {1: "Drogon", 2: "Rhaegal", 3: "Ruby",
                       4: "Toothless", 5: "Charmander"}

    if birds is None:
        birds = [hawk_id_to_name[bid] for bid in sorted(frame_info_df["BirdID"].unique())
                 if bid in hawk_id_to_name]

    components_by_bird = {}
    for bird in birds:
        if year is not None:
            bird_year = year
        else:
            # Use the most recent year for this bird (Period 2 preferred)
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
    """
    Get the sizes of the input data.
    """
    
    n_frames = pca_input.shape[0]
    n_markers = pca_input.shape[1]/3
    n_vars = pca_input.shape[1]

    return n_frames, n_markers, n_vars

def get_PCA_input(markers):
    """
    Reshape the data to be [n, nMarkers*3]
    """
    n_markers = markers.shape[1]
    pca_input = markers.reshape(-1, n_markers*3)

    return pca_input


def test_PCA_output(pca_input, principal_components, scores):
    """
    Test the shape of the PCA output.
    """
    n_frames, n_markers, n_vars = get_PCA_input_sizes(pca_input)

    assert n_vars == n_markers*3, "n_vars is not equal to n_markers*3."
    assert principal_components.shape[0] == n_vars, "principal_components is not the right shape."
    assert principal_components.shape[1] == n_vars, "principal_components is not the right shape."
    assert scores.shape[0] == n_frames, "scores first dim is not the right shape."
    assert scores.shape[1] == n_vars, "scores second dim is not the right shape."




