"""PCA score manipulation, binning, and summary statistics.

Provides helpers for concatenating PCA scores with frame metadata,
binning by horizontal distance, and computing per-bin statistics.
"""

import numpy as np
import pandas as pd

from .data_filtering import filter_by

# ------- Sign conventions -------
# Keys are 0-indexed component indices; values are the multiplier.
# PC07 (index 6) = m-folding: negated so positive = more folded.
SIGN_CONVENTIONS: dict[int, int] = {6: -1}

_SIGN_LABELS: dict[int, str] = {
    6: "PC07 negated (positive = m-folded)",
}


def apply_sign_conventions(scores, verbose=True):
    """Apply interpretive sign flips to a scores array and return a copy.

    Sign conventions ensure that each principal component has an
    intuitively positive direction (e.g. PC07 is negated so that
    positive scores correspond to more m-folded wing shapes).

    Args:
        scores: Raw PCA score array of shape ``(n_frames, n_components)``.
        verbose: If True, print which conventions were applied. Defaults to True.

    Returns:
        Copy of ``scores`` with columns listed in ``SIGN_CONVENTIONS`` negated.
    """
    out = scores.copy()
    applied = []
    for col_idx, sign in SIGN_CONVENTIONS.items():
        if col_idx < out.shape[1]:
            out[:, col_idx] *= sign
            if verbose and col_idx in _SIGN_LABELS:
                applied.append(_SIGN_LABELS[col_idx])
    if verbose and applied:
        print(f"  Sign conventions applied: {', '.join(applied)}")
    return out


# ------- Scores -------

def get_score_df(scores, frame_info_df, filter=None, size_bin=0.05):
    """Build a combined score-and-metadata DataFrame binned by horizontal distance.

    Applies sign conventions, concatenates scores with frame metadata, and
    assigns each frame to a horizontal-distance bin.

    Args:
        scores: PCA score array of shape ``(n_frames, n_components)``.
        frame_info_df: Per-frame metadata DataFrame.
        filter: Boolean mask used when computing the scores, applied to
            ``frame_info_df`` before joining. Defaults to None.
        size_bin: Width of each horizontal-distance bin in metres.
            Defaults to 0.05.

    Returns:
        Tuple of (scores_df, horzDist_bins) where scores_df is the
        combined DataFrame with a ``bins`` column and horzDist_bins is
        the list of bin label strings.
    """
    scores_df = concat_df(apply_sign_conventions(scores), frame_info_df, filter)

    scores_df, horzDist_bins = bin_by_horz_distance(scores_df, size_bin)

    return scores_df, horzDist_bins


def prepare_heatmap_mode_order(_scores_df, n_modes=9):
    """Return mode columns in thematic display order for heatmap plotting.

    PC08 (collective pitching) is moved after PC05 (counter pitching)
    to group related pitching modes together visually. Sign conventions
    (e.g. PC07 negation) are applied upstream by ``get_score_df`` /
    ``apply_sign_conventions``.

    Args:
        scores_df: Score DataFrame (not modified).
        n_modes: Number of modes to include. Defaults to 9.

    Returns:
        Ordered list of PC column name strings for heatmap display.
    """
    pc_cols = [f"PC{i:02}" for i in range(1, n_modes + 1)]
    # Move PC08 next to PC05 to group pitching modes together
    if n_modes >= 8:
        pc_cols.insert(5, pc_cols.pop(pc_cols.index("PC08")))
    return pc_cols


def get_binned_scores(scores_df, **filters):
    """Compute per-bin mean, median, and standard deviation of PCA scores.

    Applies the given filters, groups the remaining frames by horizontal-
    distance bin, and computes summary statistics across bins.

    Args:
        scores_df: Score DataFrame with a ``bins`` column and ``PC01``,
            ``PC02``, … columns.
        **filters: Keyword filter arguments forwarded to ``filter_by``.

    Returns:
        Tuple of (binned_info, mean_scores, stdev_scores, median_scores)
        where each scores object is a DataFrame indexed by distance bin.

    Raises:
        ValueError: If the mean, median, and standard deviation results
            have different numbers of bins.
    """
    filter = filter_by(scores_df, **filters)

    mean_scores   = get_mean_by_bin(scores_df[filter])
    median_scores = get_median_by_bin(scores_df[filter])
    stdev_scores  = get_stdev_by_bin(scores_df[filter])

    binned_info = get_binned_info(scores_df[filter])

    if (
        mean_scores.shape[0] != median_scores.shape[0]
        or mean_scores.shape[0] != stdev_scores.shape[0]
    ):
        msg = (
            "Mean, median, and stdev scores must have the "
            "same number of bins."
        )
        raise ValueError(msg)

    return binned_info, mean_scores, stdev_scores, median_scores


def get_score_range(scores, num_frames=30):
    """Generate a triangle-wave sweep of scores for shape-mode animation.

    Creates a forward-and-back sweep spanning ±2 standard deviations
    around the mean of each principal component, providing a standardised
    range for visualising how each mode deforms the mean shape.

    Args:
        scores: Score array of shape ``(n_frames, n_components)``.
        num_frames: Number of animation frames to generate. Defaults to 30.

    Returns:
        Score array of shape ``(num_frames, n_components)`` sweeping from
        mean - 2*std to mean + 2*std and back.
    """
    min_score = np.mean(scores, axis=0) - (2 * np.std(scores, axis=0))
    max_score = np.mean(scores, axis=0) + (2 * np.std(scores, axis=0))

    half_length = num_frames // 2 + 1
    triangle_wave = np.linspace(0, 1, half_length)
    triangle_wave = np.concatenate([triangle_wave, triangle_wave[-2:0:-1]])

    return min_score + (max_score - min_score) * triangle_wave[:, np.newaxis]



# ....... Helper functions .......

def concat_df(scores, frame_info_df, filter=None):
    """Concatenate PCA scores with frame metadata into a single DataFrame.

    Optionally applies a boolean ``filter`` to ``frame_info_df`` before
    joining, so the row counts match.

    Args:
        scores: Score array of shape ``(n_frames, n_components)``.
        frame_info_df: Frame metadata DataFrame.
        filter: Boolean mask used when computing the scores. The same mask
            is applied to ``frame_info_df`` to ensure alignment. Defaults to None.

    Returns:
        Combined DataFrame with metadata columns followed by ``PC01``,
        ``PC02``, … columns.

    Raises:
        ValueError: If ``scores`` and the filtered ``frame_info_df`` have
            different numbers of rows.
    """
    if filter is not None:
        frame_info_df = frame_info_df[filter].reset_index(drop=True)

    if scores.shape[0] != frame_info_df.shape[0]:
        msg = (
            f"Size mismatch: scores has {scores.shape[0]} rows but "
            f"frame_info_df has {frame_info_df.shape[0]} rows. "
            f"Include the filter used for PCA!"
        )
        raise ValueError(
            msg
        )

    return create_scores_info_df(scores, frame_info_df)



def create_scores_info_df(scores, frame_info_df):
    """Create a DataFrame combining PCA scores with frame metadata.

    Score columns are named ``PC01``, ``PC02``, etc. and are appended
    to ``frame_info_df``.

    Args:
        scores: Score array of shape ``(n_frames, n_components)``.
        frame_info_df: Frame metadata DataFrame with the same number of
            rows as ``scores``.

    Returns:
        Concatenated DataFrame of metadata and score columns.
    """
    num_components = scores.shape[1]
    PC_names = [f'PC{i:02}' for i in np.arange(1, num_components + 1)]

    score_df = pd.DataFrame(scores, columns=PC_names)

    return pd.concat([frame_info_df, score_df], axis=1)



def bin_by_horz_distance(df, size_bin=0.05):
    """Assign each frame to a horizontal-distance bin and add a ``bins`` column.

    Args:
        df: DataFrame containing a ``HorzDistance`` column.
        size_bin: Width of each distance bin in metres. Defaults to 0.05.

    Returns:
        Tuple of (df_with_bins, bin_labels) where ``df_with_bins`` is a copy
        of ``df`` with a ``bins`` column added and ``bin_labels`` is the list
        of bin label strings.
    """
    min_val = np.floor(df['HorzDistance'].min() / size_bin) * size_bin
    max_val = np.ceil(df['HorzDistance'].max() / size_bin) * size_bin + 2 * size_bin
    bins = np.arange(min_val, max_val, size_bin)
    bins = np.around(bins, 3)
    bin_labels = bins.astype(str).tolist()
    bin_labels.pop(0)

    df['bins'] = pd.cut(df['HorzDistance'],
                        bins,
                        right=False,
                        labels=bin_labels,
                        include_lowest=True)

    return df, bin_labels


def get_binned_info(scores_df):
    """Compute per-bin means and first values for metadata columns.

    Args:
        scores_df: Score DataFrame with a ``bins`` column.

    Returns:
        DataFrame indexed by distance bin with mean flight-dynamics
        columns and first-value metadata columns.
    """
    mean_cols = [
        'time', 'HorzDistance', 'VertDistance',
        'body_pitch', 'body_roll', 'body_yaw',
    ]
    mean_cols = [col for col in mean_cols if col in scores_df.columns]

    if mean_cols:
        binned_info = scores_df.groupby('bins', observed=True)[mean_cols].mean()
    else:
        binned_info = pd.DataFrame()

    first_cols = ['frameID', 'seqID', 'BirdID', 'PerchDistance', 'Year',
                  'Obstacle', 'IMU', 'Naive']
    first_cols = [col for col in first_cols if col in scores_df.columns]

    if 'Left' in scores_df.columns:
        first_cols.append('Left')

    if first_cols:
        other_info = scores_df.groupby('bins', observed=True)[first_cols].first()
        binned_info = pd.concat([other_info, binned_info], axis=1)

    return binned_info


def get_mean_by_bin(scores_df, col_name='PC'):
    """Compute the per-bin mean for all PC columns.

    Args:
        scores_df: Score DataFrame with a ``bins`` column and ``PC``-prefixed
            columns.
        col_name: Column name prefix used to identify score columns.
            Defaults to ``'PC'``.

    Returns:
        DataFrame of mean PC scores indexed by distance bin.
    """
    PC_columns = scores_df.filter(like=col_name)
    return PC_columns.groupby(scores_df['bins'], observed=True).mean()



def get_median_by_bin(scores_df):
    """Compute the per-bin median for all PC columns.

    Args:
        scores_df: Score DataFrame with a ``bins`` column and ``PC``-prefixed
            columns.

    Returns:
        DataFrame of median PC scores indexed by distance bin.
    """
    PC_columns = scores_df.filter(like='PC')
    return PC_columns.groupby(scores_df['bins'], observed=True).median()



def get_stdev_by_bin(scores_df) -> pd.DataFrame:
    """Compute the per-bin standard deviation for all PC columns.

    Args:
        scores_df: Score DataFrame with a ``bins`` column and ``PC``-prefixed
            columns.

    Returns:
        DataFrame of PC score standard deviations indexed by distance bin.
    """
    PC_columns = scores_df.filter(like='PC')
    return PC_columns.groupby(scores_df['bins'], observed=True).std()

