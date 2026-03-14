"""
test_pca_scores.py

Tests for the pca_scores module: get_score_df(), get_binned_scores(),
get_score_range(), and all helper functions.
Covers S09 (Comparing Individual Morphing Shape Modes),
S10.1 (Projection and Animation), and S10.3 (Shape Reconstruction Accuracy).
"""

import logging
import pytest
import numpy as np
import pandas as pd

from kinematic_morphospace.pca_scores import (
    get_score_df,
    get_binned_scores,
    get_score_range,
    concat_df,
    create_scores_info_df,
    bin_by_horz_distance,
    get_binned_info,
    get_mean_by_bin,
    get_median_by_bin,
    get_stdev_by_bin,
)

logger = logging.getLogger(__name__)


# -- Fixtures --

@pytest.fixture
def synthetic_frame_info():
    """Minimal frame_info_df with 20 rows and all expected columns.
    HorzDistance in [-10, -0.5] to fall within the current hardcoded bin range."""
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame({
        'frameID': [f'frame_{i:03d}' for i in range(n)],
        'seqID': [f'seq_{i // 5:02d}' for i in range(n)],
        'time': np.linspace(-0.5, 0.0, n),
        'HorzDistance': np.linspace(-10.0, -0.5, n),
        'VertDistance': rng.uniform(0, 1, n),
        'body_pitch': rng.uniform(-15, 5, n),
        'body_roll': rng.uniform(-5, 5, n),
        'body_yaw': rng.uniform(-10, 10, n),
        'BirdID': rng.choice([1, 2, 3], n),
        'PerchDistance': rng.choice([9, 12], n),
        'Year': rng.choice([2017, 2020], n),
        'Naive': rng.choice([0, 1], n),
        'Obstacle': rng.choice([0, 1], n),
        'IMU': rng.choice([0, 1], n),
        'Left': rng.choice([0, 1], n),
        'Turn': rng.choice(['Straight', 'Left', 'Right'], n),
    })


@pytest.fixture
def synthetic_scores():
    """Synthetic PCA scores: 20 frames, 3 components."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((20, 3))


@pytest.fixture
def binned_scores_df(synthetic_scores, synthetic_frame_info):
    """A scores_df that already has a 'bins' column, ready for aggregation tests."""
    scores_df = create_scores_info_df(synthetic_scores, synthetic_frame_info)
    scores_df, _ = bin_by_horz_distance(scores_df, size_bin=2.0)
    return scores_df


@pytest.fixture
def real_scores_and_info(sample_unilateraldata_path):
    """Run PCA on real test data, return scores + frame_info_df."""
    from kinematic_morphospace import load_data, process_data
    from kinematic_morphospace.pca_core import run_PCA
    data_csv = load_data(sample_unilateraldata_path)
    markers, _, _, frame_info_df = process_data(data_csv)
    components, scores, pca_obj = run_PCA(markers)
    return scores, frame_info_df, markers, pca_obj


# -- TestCreateScoresInfoDf --

class TestCreateScoresInfoDf:
    def test_output_is_dataframe(self, synthetic_scores, synthetic_frame_info):
        """Result should be a pandas DataFrame."""
        result = create_scores_info_df(synthetic_scores, synthetic_frame_info)
        assert isinstance(result, pd.DataFrame)

    def test_columns_include_pc_names(self, synthetic_scores, synthetic_frame_info):
        """Columns should include zero-padded PC names (PC01, PC02, PC03)
        and all original frame_info columns."""
        result = create_scores_info_df(synthetic_scores, synthetic_frame_info)
        for pc_name in ['PC01', 'PC02', 'PC03']:
            assert pc_name in result.columns, f"Missing column {pc_name}"
        for col in synthetic_frame_info.columns:
            assert col in result.columns, f"Missing frame_info column {col}"

    def test_row_count_matches_input(self, synthetic_scores, synthetic_frame_info):
        """Output should have the same number of rows as the input scores."""
        result = create_scores_info_df(synthetic_scores, synthetic_frame_info)
        assert len(result) == synthetic_scores.shape[0]


# -- TestConcatDf (Bug #2) --

class TestConcatDf:
    def test_concat_without_filter(self, synthetic_scores, synthetic_frame_info):
        """With filter=None, returns DataFrame with rows matching scores."""
        result = concat_df(synthetic_scores, synthetic_frame_info, filter=None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == synthetic_scores.shape[0]

    def test_concat_with_boolean_filter(self, synthetic_scores, synthetic_frame_info):
        """Boolean filter should select a subset of frame_info rows."""
        mask = synthetic_frame_info['BirdID'] == 1
        n_selected = mask.sum()
        # Scores must match the filtered size
        filtered_scores = synthetic_scores[:n_selected]
        result = concat_df(filtered_scores, synthetic_frame_info, filter=mask)
        assert len(result) == n_selected

    def test_size_mismatch_raises_valueerror(self, synthetic_frame_info):
        """BUG #2: Mismatched scores/frame_info sizes should raise ValueError.
        Current code has 'pass' instead of raising."""
        bad_scores = np.zeros((15, 3))  # 15 rows vs 20 in frame_info
        with pytest.raises(ValueError, match="[Ss]ize"):
            concat_df(bad_scores, synthetic_frame_info, filter=None)

    def test_size_mismatch_with_wrong_filter_raises(self, synthetic_scores, synthetic_frame_info):
        """BUG #2: Filter that produces wrong size should raise ValueError."""
        # Filter reduces frame_info to ~7 rows, but scores has 20
        mask = synthetic_frame_info['BirdID'] == 1
        with pytest.raises(ValueError, match="[Ss]ize"):
            concat_df(synthetic_scores, synthetic_frame_info, filter=mask)

    def test_filter_resets_index(self, synthetic_frame_info):
        """After applying a filter, the index should be contiguous."""
        mask = synthetic_frame_info['BirdID'] == 1
        n_selected = mask.sum()
        filtered_scores = np.zeros((n_selected, 3))
        result = concat_df(filtered_scores, synthetic_frame_info, filter=mask)
        assert list(result.index) == list(range(len(result)))


# -- TestBinByHorzDistance (Bug #4) --

class TestBinByHorzDistance:
    def test_adds_bins_column(self, synthetic_frame_info):
        """Output DataFrame should have a 'bins' column."""
        df, _ = bin_by_horz_distance(synthetic_frame_info.copy(), size_bin=0.5)
        assert 'bins' in df.columns

    def test_returns_bin_labels(self, synthetic_frame_info):
        """Second return value should be a list of string bin labels."""
        _, labels = bin_by_horz_distance(synthetic_frame_info.copy(), size_bin=0.5)
        assert isinstance(labels, list)
        assert all(isinstance(lbl, str) for lbl in labels)

    def test_all_rows_binned_within_range(self, synthetic_frame_info):
        """When HorzDistance values all fall within [-12.2, 0.2),
        no NaN should appear in the bins column."""
        df, _ = bin_by_horz_distance(synthetic_frame_info.copy(), size_bin=0.5)
        assert df['bins'].isna().sum() == 0, \
            "No rows should have NaN bins when data is within range"

    def test_out_of_range_data_gets_valid_bins(self):
        """BUG #4: Data outside the hardcoded [-12.2, 0.2) range should
        still receive valid bins (not NaN)."""
        df = pd.DataFrame({'HorzDistance': [5.0, 10.0, -15.0]})
        result, _ = bin_by_horz_distance(df, size_bin=1.0)
        assert result['bins'].isna().sum() == 0, \
            "All data should be binned, regardless of range"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_positive_horzdist_gets_binned(self, real_scores_and_info):
        """BUG #4: Real test data has positive HorzDistance (0.3-11.5).
        All rows should receive valid bins."""
        _, frame_info_df, _, _ = real_scores_and_info
        df = frame_info_df.copy()
        result, _ = bin_by_horz_distance(df, size_bin=0.5)
        assert result['bins'].isna().sum() == 0, \
            f"Positive HorzDistance data should be binned. " \
            f"Range: [{df['HorzDistance'].min():.1f}, {df['HorzDistance'].max():.1f}]"

    def test_custom_bin_size(self, synthetic_frame_info):
        """With size_bin=2.0, the number of bins should be based on range/size_bin."""
        df, labels = bin_by_horz_distance(synthetic_frame_info.copy(), size_bin=2.0)
        assert len(labels) > 0
        assert 'bins' in df.columns


# -- TestGetBinnedInfo --

class TestGetBinnedInfo:
    def test_returns_dataframe(self, binned_scores_df):
        """Output should be a pandas DataFrame."""
        result = get_binned_info(binned_scores_df)
        assert isinstance(result, pd.DataFrame)

    def test_mean_columns_present(self, binned_scores_df):
        """Output should contain mean-aggregated numeric columns."""
        result = get_binned_info(binned_scores_df)
        for col in ['time', 'HorzDistance', 'VertDistance', 'body_pitch']:
            assert col in result.columns, f"Missing mean column {col}"

    def test_first_columns_present(self, binned_scores_df):
        """Output should contain first-value categorical columns."""
        result = get_binned_info(binned_scores_df)
        for col in ['frameID', 'seqID', 'BirdID']:
            assert col in result.columns, f"Missing first-value column {col}"


# -- TestGetMeanByBin --

class TestGetMeanByBin:
    def test_output_shape(self, binned_scores_df):
        """Grouped mean DataFrame should have PC columns and rows per non-empty bin."""
        result = get_mean_by_bin(binned_scores_df)
        assert isinstance(result, pd.DataFrame)
        pc_cols = [c for c in result.columns if c.startswith('PC')]
        assert len(pc_cols) == 3  # PC01, PC02, PC03

    def test_mean_values_correct(self):
        """For known data, verify computed means match expected values."""
        df = pd.DataFrame({
            'PC01': [1.0, 3.0, 10.0, 20.0],
            'PC02': [2.0, 4.0, 20.0, 40.0],
            'bins': pd.Categorical(['a', 'a', 'b', 'b']),
        })
        result = get_mean_by_bin(df)
        np.testing.assert_allclose(result.loc['a', 'PC01'], 2.0)
        np.testing.assert_allclose(result.loc['a', 'PC02'], 3.0)
        np.testing.assert_allclose(result.loc['b', 'PC01'], 15.0)
        np.testing.assert_allclose(result.loc['b', 'PC02'], 30.0)


# -- TestGetMedianByBin --

class TestGetMedianByBin:
    def test_output_shape(self, binned_scores_df):
        """Grouped median DataFrame should have PC columns."""
        result = get_median_by_bin(binned_scores_df)
        assert isinstance(result, pd.DataFrame)
        pc_cols = [c for c in result.columns if c.startswith('PC')]
        assert len(pc_cols) == 3

    def test_median_values_correct(self):
        """For known data, verify computed medians."""
        df = pd.DataFrame({
            'PC01': [1.0, 3.0, 5.0, 10.0, 20.0, 30.0],
            'bins': pd.Categorical(['a', 'a', 'a', 'b', 'b', 'b']),
        })
        result = get_median_by_bin(df)
        np.testing.assert_allclose(result.loc['a', 'PC01'], 3.0)
        np.testing.assert_allclose(result.loc['b', 'PC01'], 20.0)


# -- TestGetStdevByBin --

class TestGetStdevByBin:
    def test_output_shape(self, binned_scores_df):
        """Grouped stdev DataFrame should have PC columns."""
        result = get_stdev_by_bin(binned_scores_df)
        assert isinstance(result, pd.DataFrame)
        pc_cols = [c for c in result.columns if c.startswith('PC')]
        assert len(pc_cols) == 3

    def test_single_frame_per_bin_returns_nan(self):
        """With only 1 frame in a bin, stdev should be NaN (pandas std default)."""
        df = pd.DataFrame({
            'PC01': [5.0],
            'bins': pd.Categorical(['a']),
        })
        result = get_stdev_by_bin(df)
        assert result.loc['a', 'PC01'] != result.loc['a', 'PC01']  # NaN check


# -- TestGetScoreDf (integration) --

class TestGetScoreDf:
    def test_returns_tuple_of_df_and_labels(self, synthetic_scores, synthetic_frame_info):
        """Should return (DataFrame, list)."""
        result = get_score_df(synthetic_scores, synthetic_frame_info)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

    def test_output_df_has_bins_and_pc_columns(self, synthetic_scores, synthetic_frame_info):
        """The DataFrame should have both 'bins' and PC01/PC02/... columns."""
        df, _ = get_score_df(synthetic_scores, synthetic_frame_info)
        assert 'bins' in df.columns
        for pc in ['PC01', 'PC02', 'PC03']:
            assert pc in df.columns


# -- TestGetBinnedScores (integration) --

class TestGetBinnedScores:
    def test_returns_four_outputs(self, binned_scores_df):
        """Should return (binned_info, mean_scores, stdev_scores, median_scores)."""
        result = get_binned_scores(binned_scores_df)
        assert len(result) == 4

    def test_output_shapes_consistent(self, binned_scores_df):
        """Mean, median, and stdev DataFrames should have the same shape."""
        _, mean_scores, stdev_scores, median_scores = get_binned_scores(binned_scores_df)
        assert mean_scores.shape == median_scores.shape
        assert mean_scores.shape == stdev_scores.shape


# -- TestGetScoreRange (S10.1) --

class TestGetScoreRange:
    def test_output_shape(self, synthetic_scores):
        """With num_frames=30, output should be (30, n_components)."""
        result = get_score_range(synthetic_scores, num_frames=30)
        assert result.shape == (30, synthetic_scores.shape[1])

    def test_values_within_two_std(self, synthetic_scores):
        """All output values should be within [mean-2*std, mean+2*std]."""
        result = get_score_range(synthetic_scores, num_frames=30)
        mean = np.mean(synthetic_scores, axis=0)
        std = np.std(synthetic_scores, axis=0)
        lower = mean - 2 * std
        upper = mean + 2 * std
        for i in range(synthetic_scores.shape[1]):
            assert np.all(result[:, i] >= lower[i] - 1e-10), \
                f"Component {i}: values below lower bound"
            assert np.all(result[:, i] <= upper[i] + 1e-10), \
                f"Component {i}: values above upper bound"

    def test_triangle_wave_symmetry(self, synthetic_scores):
        """First frame should be at min (mean-2std), peak frame at max (mean+2std),
        and the wave should be monotonically increasing then decreasing."""
        result = get_score_range(synthetic_scores, num_frames=30)
        mean = np.mean(synthetic_scores, axis=0)
        std = np.std(synthetic_scores, axis=0)
        expected_min = mean - 2 * std
        expected_max = mean + 2 * std
        # First frame should be at the minimum
        np.testing.assert_allclose(result[0], expected_min, atol=1e-10,
            err_msg="First frame should be at mean - 2*std")
        # Peak frame at half_length - 1
        half_length = 30 // 2 + 1  # 16
        np.testing.assert_allclose(result[half_length - 1], expected_max, atol=1e-10,
            err_msg="Peak frame should be at mean + 2*std")
        # First half should be monotonically increasing (component 0)
        first_half = result[:half_length, 0]
        assert np.all(np.diff(first_half) >= -1e-10), \
            "First half should be monotonically increasing"
