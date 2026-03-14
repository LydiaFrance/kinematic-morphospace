"""
test_pca_core.py

Tests for the pca_core module: run_PCA(), run_PCA_birds(),
get_PCA_input(), get_PCA_input_sizes(), test_PCA_output().
Exposes known issue #5 (PCA non-determinism) and AssertionError typo.
"""

import logging
import warnings
import pytest
import numpy as np

from kinematic_morphospace import load_data, process_data
from kinematic_morphospace.pca_core import (
    run_PCA,
    run_PCA_birds,
    get_PCA_input,
    get_PCA_input_sizes,
)
# Alias to avoid pytest collecting this as a test
from kinematic_morphospace.pca_core import test_PCA_output as validate_pca_output

logger = logging.getLogger(__name__)


# -- Fixtures --

@pytest.fixture
def sample_markers(sample_unilateraldata_path):
    """Load real test data and return markers array (n_frames, n_markers, 3)."""
    data_csv = load_data(sample_unilateraldata_path)
    markers, _, _, _ = process_data(data_csv)
    return markers


@pytest.fixture
def sample_frame_info_df(sample_unilateraldata_path):
    """Load real test data and return frame_info_df."""
    data_csv = load_data(sample_unilateraldata_path)
    _, _, _, frame_info_df = process_data(data_csv)
    return frame_info_df


@pytest.fixture
def synthetic_markers():
    """Small synthetic marker data for fast tests."""
    rng = np.random.default_rng(42)
    # 100 frames, 8 markers, 3 coordinates
    return rng.standard_normal((100, 8, 3))


# -- get_PCA_input tests --

class TestGetPCAInput:
    def test_reshapes_correctly(self, synthetic_markers):
        result = get_PCA_input(synthetic_markers)
        assert result.shape == (100, 24)

    def test_preserves_data(self, synthetic_markers):
        result = get_PCA_input(synthetic_markers)
        # First marker's x,y,z should be the first 3 values of each row
        np.testing.assert_array_equal(
            result[0, :3],
            synthetic_markers[0, 0, :]
        )

    def test_different_marker_counts(self):
        markers = np.zeros((50, 4, 3))
        result = get_PCA_input(markers)
        assert result.shape == (50, 12)


# -- get_PCA_input_sizes tests --

class TestGetPCAInputSizes:
    def test_returns_correct_sizes(self):
        pca_input = np.zeros((100, 24))
        n_frames, n_markers, n_vars = get_PCA_input_sizes(pca_input)
        assert n_frames == 100
        assert n_markers == 8.0
        assert n_vars == 24


# -- run_PCA tests --

class TestRunPCA:
    def test_output_shapes(self, synthetic_markers):
        components, scores, pca_obj = run_PCA(synthetic_markers)
        n_frames, n_markers, _ = synthetic_markers.shape
        n_vars = n_markers * 3
        assert components.shape == (n_vars, n_vars)
        assert scores.shape == (n_frames, n_vars)

    def test_variance_explained_sums_to_one(self, synthetic_markers):
        _, _, pca_obj = run_PCA(synthetic_markers)
        total_var = pca_obj.explained_variance_ratio_.sum()
        np.testing.assert_allclose(total_var, 1.0, atol=1e-10)

    def test_deterministic_results(self, synthetic_markers):
        """Known issue #5: PCA() without random_state may be non-deterministic.
        For our data sizes this should still pass, but the fix is to add random_state."""
        components1, scores1, _ = run_PCA(synthetic_markers)
        components2, scores2, _ = run_PCA(synthetic_markers)
        np.testing.assert_array_equal(components1, components2)
        np.testing.assert_array_equal(scores1, scores2)

    def test_projection_onto_different_data(self, synthetic_markers):
        """Fit PCA on subset, project full data."""
        subset = synthetic_markers[:50]
        components, scores, pca_obj = run_PCA(subset, project_data=synthetic_markers)
        assert components.shape == (24, 24)
        assert scores.shape == (100, 24)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_on_real_data(self, sample_markers):
        n_frames, n_markers, _ = sample_markers.shape
        n_vars = n_markers * 3
        components, scores, pca_obj = run_PCA(sample_markers)
        assert components.shape == (n_vars, n_vars)
        assert scores.shape == (n_frames, n_vars)
        # First few components should capture most variance
        cumvar = np.cumsum(pca_obj.explained_variance_ratio_)
        assert cumvar[3] > 0.9, "First 4 PCs should explain >90% variance on real data"

    def test_mean_shape_stored(self, synthetic_markers):
        """PCA object should store the mean for reconstruction."""
        _, _, pca_obj = run_PCA(synthetic_markers)
        assert hasattr(pca_obj, 'mean_')
        assert pca_obj.mean_.shape == (24,)


# -- run_PCA_birds tests --

class TestRunPCABirds:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_derives_birds_and_years_from_data(self, sample_markers, sample_frame_info_df):
        """run_PCA_birds now derives birds and years from data, so it should
        not crash even when the test data only contains a subset of birds."""
        result = run_PCA_birds(sample_markers, sample_frame_info_df, filter_on=False)
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_single_bird_pca(self, sample_markers, sample_frame_info_df):
        """Verify PCA works on a single bird's data extracted manually."""
        from kinematic_morphospace.data_filtering import filter_by
        mask = filter_by(sample_frame_info_df, hawkname="Drogon", year=2017)
        if mask.sum() > 0:
            bird_markers = sample_markers[mask]
            components, scores, pca_obj = run_PCA(bird_markers)
            n_vars = sample_markers.shape[1] * 3
            assert components.shape == (n_vars, n_vars)


# -- validate_pca_output (test_PCA_output) tests --

class TestPCAOutputValidation:
    def test_valid_output_passes(self):
        pca_input = np.zeros((100, 24))
        components = np.zeros((24, 24))
        scores = np.zeros((100, 24))
        # Should not raise
        validate_pca_output(pca_input, components, scores)

    def test_mismatched_components_raises(self):
        pca_input = np.zeros((100, 24))
        components = np.zeros((12, 24))  # Wrong first dim
        scores = np.zeros((100, 24))
        with pytest.raises(AssertionError):
            validate_pca_output(pca_input, components, scores)

    def test_mismatched_scores_raises(self):
        pca_input = np.zeros((100, 24))
        components = np.zeros((24, 24))
        scores = np.zeros((50, 24))  # Wrong first dim
        with pytest.raises(AssertionError):
            validate_pca_output(pca_input, components, scores)
