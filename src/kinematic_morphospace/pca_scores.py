"""PCA score manipulation, binning, and summary statistics.

Provides helpers for concatenating PCA scores with frame metadata,
binning by horizontal distance, and computing per-bin statistics.
"""

import numpy as np
import pandas as pd

from .data_filtering import filter_by


# ------- Scores -------

def get_score_df(scores, frame_info_df, filter=None, size_bin=0.05):

    scores_df = concat_df(scores, frame_info_df, filter)

    scores_df, horzDist_bins = bin_by_horz_distance(scores_df, size_bin)

    return scores_df, horzDist_bins

def get_binned_scores(scores_df, **filters):

    filter = filter_by(scores_df, **filters)

    mean_scores   = get_mean_by_bin(scores_df[filter])
    median_scores = get_median_by_bin(scores_df[filter])
    stdev_scores  = get_stdev_by_bin(scores_df[filter])

    binned_info = get_binned_info(scores_df[filter])

    # Check each of the outputs are the same size
    if mean_scores.shape[0] != median_scores.shape[0] or mean_scores.shape[0] != stdev_scores.shape[0]:
        raise ValueError("Mean, median, and stdev scores must have the same number of bins.")
    

    # Find any index where the stdev is Nan
    stdev_nan = stdev_scores.isnull().any(axis=1)


    return binned_info, mean_scores, stdev_scores, median_scores


def get_score_range(scores, num_frames=30):
    """Generate a triangle-wave sweep of scores for animation.

    Creates a forward-and-back sweep spanning +/- 2 standard deviations
    around the mean of each principal component.

    Parameters
    ----------
    scores : np.ndarray
        Score array of shape ``(n_frames, n_components)``.
    num_frames : int, optional
        Number of animation frames to generate.

    Returns
    -------
    np.ndarray
        Score array of shape ``(num_frames, n_components)``.
    """
    num_components = scores.shape[1]

    min_score = np.mean(scores, axis=0) - (2 * np.std(scores, axis=0))
    max_score = np.mean(scores, axis=0) + (2 * np.std(scores, axis=0))

    # Create a triangle wave for the time series
    half_length = num_frames // 2 + 1
    triangle_wave = np.linspace(0, 1, half_length)
    triangle_wave = np.concatenate([triangle_wave, triangle_wave[-2:0:-1]])

    score_frames = min_score + (max_score - min_score) * triangle_wave[:, np.newaxis]

    # # Initialize score_frames with the shape [num_frames, num_components]
    # score_frames = np.zeros([num_frames, num_components])

    # for ii in range(num_components):
    #     # Create forward and backward ranges for each component using np.linspace
    #     forward = np.linspace(min_score[ii], max_score[ii], num=half_length)
    #     backward = forward[::-1]  # Reverse the forward range

    #     # Combine forward and backward, and assign to the i-th column
    #     score_frames[:, ii] = np.concatenate((forward, backward[:num_frames - half_length]))

    return score_frames


# ....... Helper functions .......

def concat_df(scores, frame_info_df, filter=None):
    """Concatenate PCA scores with frame metadata into a single DataFrame.

    Optionally applies a boolean *filter* to *frame_info_df* before
    joining, so the row counts match.

    Parameters
    ----------
    scores : np.ndarray
        Score array of shape ``(n_frames, n_components)``.
    frame_info_df : pd.DataFrame
        Frame metadata DataFrame.
    filter : np.ndarray or pd.Series, optional
        Boolean mask used when computing the scores. The same mask is
        applied to *frame_info_df* to ensure alignment.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with metadata columns and ``PC01``, ``PC02``, ...
    """
    # Filter the info dataframes to match how the scores were calculated
    if filter is not None:
        frame_info_df = frame_info_df[filter].reset_index(drop=True)

    # Check this matches the sizes
    if scores.shape[0] != frame_info_df.shape[0]:
        raise ValueError(
            f"Size mismatch: scores has {scores.shape[0]} rows but "
            f"frame_info_df has {frame_info_df.shape[0]} rows. "
            f"Include the filter used for PCA!"
        )

    scores_df = create_scores_info_df(scores, frame_info_df)

    return scores_df

def create_scores_info_df(scores, frame_info_df):
    """Create a DataFrame combining PCA scores with frame metadata.

    Score columns are named ``PC01``, ``PC02``, etc. and are appended
    to *frame_info_df*.

    Parameters
    ----------
    scores : np.ndarray
        Score array of shape ``(n_frames, n_components)``.
    frame_info_df : pd.DataFrame
        Frame metadata DataFrame with the same number of rows as *scores*.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of metadata and score columns.
    """
    # Add the scores to the dataframe. Give the column the name 'PC1' etc.
    num_components = scores.shape[1]
    PC_names = [f'PC{i:02}' for i in np.arange(1, num_components+1)]
    
    # ["PC" + str(ii) for ii in range(1, num_components+1)]

    # Create a pandas dataframe
    score_df = pd.DataFrame(scores, columns=PC_names)

    scores_df = pd.concat([frame_info_df, score_df], axis=1)

    return scores_df

def bin_by_horz_distance(df, size_bin=0.05):

    min_val = np.floor(df['HorzDistance'].min() / size_bin) * size_bin
    max_val = np.ceil(df['HorzDistance'].max() / size_bin) * size_bin + 2 * size_bin
    bins = np.arange(min_val, max_val, size_bin)
    bins = np.around(bins, 3)
    bin_labels = bins.astype(str).tolist()
    # make label one smaller
    bin_labels.pop(0)

    df['bins'] = pd.cut(df['HorzDistance'], 
                                bins, 
                                right=False, 
                                labels = bin_labels, 
                                include_lowest=True)

    # Remove any rows with NaN values
    # scores_df = scores_df.dropna()

    return df, bin_labels

def get_binned_info(scores_df):
    # Define potential columns for mean calculation
    mean_cols = ['time', 'HorzDistance', 'VertDistance', 'body_pitch', 'body_roll', 'body_yaw']
    # Filter to only include columns that exist in the dataframe
    mean_cols = [col for col in mean_cols if col in scores_df.columns]
    
    # Only calculate means if we have columns to calculate them for
    if mean_cols:
        binned_info = scores_df.groupby('bins', observed=True)[mean_cols].mean()
    else:
        binned_info = pd.DataFrame()

    # Define potential columns for first value selection
    first_cols = ['frameID', 'seqID', 'BirdID', 'PerchDistance', 'Year', 
                 'Obstacle', 'IMU', 'Naive']
    # Filter to only include columns that exist in the dataframe
    first_cols = [col for col in first_cols if col in scores_df.columns]
    
    # Add 'Left' if it exists
    if 'Left' in scores_df.columns:
        first_cols.append('Left')

    # Only get first values if we have columns to get them for
    if first_cols:
        other_info = scores_df.groupby('bins', observed=True)[first_cols].first()
        binned_info = pd.concat([other_info, binned_info], axis=1)

    return binned_info


def get_mean_by_bin(scores_df, col_name='PC'):

    # Get the PC names
    PC_columns = scores_df.filter(like=col_name)


    # observed=True is to prevent a warning about future behaviour. 
    # See https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-specify
    
    mean_scores = PC_columns.groupby(scores_df['bins'], observed=True).mean()

    return mean_scores

def get_median_by_bin(scores_df):

    # Get the PC names
    PC_columns = scores_df.filter(like='PC')


    # observed=True is to prevent a warning about future behaviour. 
    # See https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-specify
    
    median_scores = PC_columns.groupby(scores_df['bins'], observed=True).median()

    return median_scores


def get_stdev_by_bin(scores_df):

    # Get the PC names
    PC_columns = scores_df.filter(like='PC')

    # observed=True is to prevent a warning about future behaviour. 
    # See https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-specify
    
    stdev_scores = PC_columns.groupby(scores_df['bins'], observed=True).std()


    return stdev_scores

