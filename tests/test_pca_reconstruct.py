"""
test_pca_reconstruct.py

Tests for the pca_reconstruct module: reconstruct().
Covers S05.2 (Projection to Symmetrical Components),
S06 (Morphing Shape Modes — reconstruction accuracy),
and S07 (Before vs After Rotation — PCA comparison via reconstruction).
"""

import logging
import pytest
import numpy as np

from kinematic_morphospace.pca_reconstruct import reconstruct

logger = logging.getLogger(__name__)


# -- Fixtures --

@pytest.fixture
def pca_results():
    """Run PCA on synthetic data and return components, scores, pca_obj, markers."""
    from kinematic_morphospace.pca_core import run_PCA
    rng = np.random.default_rng(42)
    markers = rng.standard_normal((100, 4, 3))  # 4 markers, 3 coords
    components, scores, pca_obj = run_PCA(markers)
    return components, scores, pca_obj, markers


@pytest.fixture
def real_pca_results(sample_unilateraldata_path):
    """Run PCA on real unilateral test data."""
    from kinematic_morphospace import load_data, process_data
    from kinematic_morphospace.pca_core import run_PCA
    data_csv = load_data(sample_unilateraldata_path)
    markers, _, _, _ = process_data(data_csv)
    components, scores, pca_obj = run_PCA(markers)
    return components, scores, pca_obj, markers


# -- Basic reconstruction tests --

class TestReconstructBasic:
    def test_output_shape(self, pca_results):
        """Reconstructed frames should have (n_frames, n_markers, 3)."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        result = reconstruct(scores, components, mu)
        assert result.shape == markers.shape

    def test_full_reconstruction_matches_original(self, pca_results):
        """Using all components should perfectly reconstruct the original data."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        result = reconstruct(scores, components, mu)
        np.testing.assert_allclose(
            result, markers, atol=1e-10,
            err_msg="Full reconstruction should match original data"
        )

    def test_partial_reconstruction_shape(self, pca_results):
        """Using a subset of components should return correct shape."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        result = reconstruct(scores, components, mu, components_list=[0, 1])
        assert result.shape == markers.shape

    def test_partial_reconstruction_loses_detail(self, pca_results):
        """Partial reconstruction should NOT exactly match original."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        result = reconstruct(scores, components, mu, components_list=[0])
        # Should not be identical (unless data is rank-1)
        assert not np.allclose(result, markers, atol=1e-5)

    def test_more_components_better_reconstruction(self, pca_results):
        """More components should give a better reconstruction (lower error)."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)

        recon_1 = reconstruct(scores, components, mu, components_list=[0])
        recon_2 = reconstruct(scores, components, mu, components_list=[0, 1])
        recon_4 = reconstruct(scores, components, mu, components_list=[0, 1, 2, 3])

        error_1 = np.mean((recon_1 - markers) ** 2)
        error_2 = np.mean((recon_2 - markers) ** 2)
        error_4 = np.mean((recon_4 - markers) ** 2)

        assert error_1 > error_2, "2 components should be better than 1"
        assert error_2 > error_4, "4 components should be better than 2"

    def test_zero_scores_returns_mean(self, pca_results):
        """If all scores are zero, reconstruction should equal the mean shape."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        zero_scores = np.zeros_like(scores)
        result = reconstruct(zero_scores, components, mu)
        # Every frame should be the mean
        for i in range(result.shape[0]):
            np.testing.assert_allclose(result[i], mu[0], atol=1e-12)

    def test_components_list_defaults_to_all(self, pca_results):
        """None components_list should use all components (same as explicit list)."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        result_default = reconstruct(scores, components, mu, components_list=None)
        all_comps = list(range(components.shape[1]))
        result_explicit = reconstruct(scores, components, mu, components_list=all_comps)
        np.testing.assert_array_equal(result_default, result_explicit)


# -- Input validation tests --

class TestReconstructValidation:
    def test_score_frames_not_array_raises_typeerror(self, pca_results):
        components, _, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        with pytest.raises(TypeError, match="numpy array"):
            reconstruct([[1, 2, 3]], components, mu)

    def test_score_frames_not_2d_raises_valueerror(self, pca_results):
        components, _, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        with pytest.raises(ValueError, match="2d"):
            reconstruct(np.zeros((10,)), components, mu)

    def test_score_frames_3d_raises_valueerror(self, pca_results):
        components, _, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        with pytest.raises(ValueError, match="2d"):
            reconstruct(np.zeros((10, 5, 3)), components, mu)

    def test_mismatched_scores_components_raises(self, pca_results):
        """scores columns must match components rows."""
        components, _, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        bad_scores = np.zeros((10, components.shape[0] + 5))  # wrong columns
        with pytest.raises(AssertionError):
            reconstruct(bad_scores, components, mu)

    def test_mu_not_3d_raises(self, pca_results):
        components, scores, _, markers = pca_results
        bad_mu = np.zeros((markers.shape[1], 3))  # 2D instead of 3D
        with pytest.raises(AssertionError):
            reconstruct(scores, components, bad_mu)


# -- Real data tests --

class TestReconstructRealData:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_full_reconstruction_on_real_data(self, real_pca_results):
        """Full reconstruction should match original real data."""
        components, scores, pca_obj, markers = real_pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        result = reconstruct(scores, components, mu)
        np.testing.assert_allclose(result, markers, atol=1e-8)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_first_4_modes_explain_most_variance(self, real_pca_results):
        """Reconstruction with first 4 modes should capture >90% of the
        shape (spec: CEV4 ≈ 0.96 for real hawk data)."""
        components, scores, pca_obj, markers = real_pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)

        recon_4 = reconstruct(scores, components, mu, components_list=[0, 1, 2, 3])
        recon_all = reconstruct(scores, components, mu)

        # Calculate reconstruction error relative to total variance
        total_error = np.sum((recon_all - markers) ** 2)
        partial_error = np.sum((recon_4 - markers) ** 2)

        # Alternatively, check CEV from pca object
        cev4 = np.sum(pca_obj.explained_variance_ratio_[:4])
        logger.info(f"CEV4 on test data: {cev4:.4f}")
        assert cev4 > 0.95, f"First 4 PCs should explain >95% variance, got {cev4:.4f}"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_symmetric_projection_reconstruction(self, real_pca_results):
        """S05.2: Reconstruct using only first 2 components (symmetric projection).
        This is the key preprocessing step for rotation correction."""
        components, scores, pca_obj, markers = real_pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)

        sym_recon = reconstruct(scores, components, mu, components_list=[0, 1])
        assert sym_recon.shape == markers.shape

        # Symmetric projection should lose detail but preserve overall structure
        full_recon = reconstruct(scores, components, mu)
        sym_error = np.mean((sym_recon - full_recon) ** 2)
        assert sym_error > 0, "Symmetric projection should differ from full reconstruction"


# -- Animation tests (S10.1) --

class TestReconstructAnimation:
    """S10.1: Integration of get_score_range() with reconstruct() for animations."""

    def test_animation_frames_output_shape(self, pca_results):
        """Score range → reconstruct should produce (num_frames, n_markers, 3)."""
        from kinematic_morphospace.pca_scores import get_score_range
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        score_range = get_score_range(scores, num_frames=30)
        result = reconstruct(score_range, components, mu)
        assert result.shape == (30, markers.shape[1], 3)

    def test_single_component_animation(self, pca_results):
        """Animation for a single component should produce valid output
        that varies across frames."""
        from kinematic_morphospace.pca_scores import get_score_range
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        score_range = get_score_range(scores, num_frames=20)
        result = reconstruct(score_range, components, mu, components_list=[0])
        assert result.shape == (20, markers.shape[1], 3)
        # Frames should not all be identical (the mode varies)
        assert not np.allclose(result[0], result[10])

    def test_animation_mean_frame_near_data_mean(self, pca_results):
        """At the midpoint of the wave (peak), the reconstructed frame
        should differ from the mean shape."""
        from kinematic_morphospace.pca_scores import get_score_range
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        score_range = get_score_range(scores, num_frames=30)
        result = reconstruct(score_range, components, mu)
        # First frame (min of wave) should differ from peak frame
        half_length = 30 // 2 + 1
        assert not np.allclose(result[0], result[half_length - 1])

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_animation_on_real_data(self, real_pca_results):
        """Animation pipeline should work on real data."""
        from kinematic_morphospace.pca_scores import get_score_range
        components, scores, pca_obj, markers = real_pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        score_range = get_score_range(scores, num_frames=30)
        result = reconstruct(score_range, components, mu)
        assert result.shape == (30, markers.shape[1], 3)
        assert np.all(np.isfinite(result))


# -- RMSE tests (S10.3) --

class TestReconstructRMSE:
    """S10.3: Reconstruction error as RMSE."""

    def test_full_reconstruction_rmse_zero(self, pca_results):
        """Full reconstruction RMSE should be ~0."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        result = reconstruct(scores, components, mu)
        rmse = np.sqrt(np.mean((result - markers) ** 2))
        assert rmse < 1e-10, f"Full reconstruction RMSE should be ~0, got {rmse}"

    def test_partial_rmse_decreases_with_more_components(self, pca_results):
        """RMSE should decrease as more components are used."""
        components, scores, pca_obj, markers = pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)

        rmse_1 = np.sqrt(np.mean(
            (reconstruct(scores, components, mu, components_list=[0]) - markers) ** 2))
        rmse_2 = np.sqrt(np.mean(
            (reconstruct(scores, components, mu, components_list=[0, 1]) - markers) ** 2))
        rmse_4 = np.sqrt(np.mean(
            (reconstruct(scores, components, mu, components_list=[0, 1, 2, 3]) - markers) ** 2))

        assert rmse_1 > rmse_2, f"2 PCs should be better than 1: {rmse_1} vs {rmse_2}"
        assert rmse_2 > rmse_4, f"4 PCs should be better than 2: {rmse_2} vs {rmse_4}"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_rmse_as_proportion_of_wingspan(self, real_pca_results, sample_wingspan_path):
        """S10.3: With 4 components, RMSE should be <2% of wingspan.
        Data is normalised by wingspan, so RMSE < 0.015 for the full
        dataset.  The test subset (4 markers, limited frames) gives
        RMSE ≈ 0.018, so the threshold is set to 0.020."""
        components, scores, pca_obj, markers = real_pca_results
        mu = pca_obj.mean_.reshape(1, markers.shape[1], 3)
        recon_4 = reconstruct(scores, components, mu, components_list=[0, 1, 2, 3])
        rmse = np.sqrt(np.mean((recon_4 - markers) ** 2))
        logger.info(f"RMSE with 4 PCs (proportion of wingspan): {rmse:.6f}")
        assert rmse < 0.020, \
            f"RMSE with 4 PCs should be <2% of wingspan, got {rmse:.6f}"
