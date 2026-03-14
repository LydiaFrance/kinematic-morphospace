"""
test_labelling.py

Tests for the labelling module: lower_dim_projection(), lower_dim_reconstruction(),
calculate_reconstruction_errors(), calculate_marker_thresholds(),
filter_low_error_frames(), clustering_analysis(), kmeans_clustering(),
analyse_clusters(), and generate_knock_out_representations().
Covers S13 (Continuum Evidence) and S14 (Individual Comparisons).
"""

import logging
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
from sklearn.decomposition import PCA

from kinematic_morphospace.labelling import (
    lower_dim_projection,
    lower_dim_reconstruction,
    calculate_reconstruction_errors,
    calculate_marker_thresholds,
    filter_low_error_frames,
    clustering_analysis,
    kmeans_clustering,
    analyse_clusters,
    generate_knock_out_representations,
)

logger = logging.getLogger(__name__)


# -- Fixtures --

@pytest.fixture
def synthetic_markers_3d():
    """Shape (100, 4, 3), rng(42)."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(100, 4, 3))


@pytest.fixture
def pca_model(synthetic_markers_3d):
    """PCA model fit on reshaped synthetic markers."""
    flat = synthetic_markers_3d.reshape(100, -1)  # (100, 12)
    model = PCA(random_state=0)
    model.fit(flat)
    return model


@pytest.fixture
def per_marker_errors_fixture():
    """Controlled (100, 4) errors with known outliers.
    First 90 rows: low errors [0, 1). Last 10 rows: high errors [5, 10)."""
    rng = np.random.default_rng(42)
    low = rng.uniform(0, 1, size=(90, 4))
    high = rng.uniform(5, 10, size=(10, 4))
    return np.vstack([low, high])


@pytest.fixture
def real_pca_results(sample_unilateraldata_path):
    """Full pipeline on real test data for integration tests."""
    from kinematic_morphospace import load_data, process_data
    from kinematic_morphospace.pca_core import run_PCA
    data_csv = load_data(sample_unilateraldata_path)
    markers, _, _, _ = process_data(data_csv)
    _, _, pca_obj = run_PCA(markers)
    return markers, pca_obj


# -- TestLowerDimProjection --

class TestLowerDimProjection:
    """Tests for lower_dim_projection()."""

    def test_output_shape(self, synthetic_markers_3d, pca_model):
        result = lower_dim_projection(synthetic_markers_3d, pca_model, n_components=4)
        n_full_features = pca_model.components_.shape[1]  # 12
        assert result.shape == (100, n_full_features)

    def test_zero_padded_beyond_n_components(self, synthetic_markers_3d, pca_model):
        n_components = 3
        result = lower_dim_projection(synthetic_markers_3d, pca_model, n_components=n_components)
        # Columns beyond n_components should be exactly zero
        assert np.all(result[:, n_components:] == 0)

    def test_first_n_components_nonzero(self, synthetic_markers_3d, pca_model):
        n_components = 4
        result = lower_dim_projection(synthetic_markers_3d, pca_model, n_components=n_components)
        # At least some values in first n_components columns should be non-zero
        assert np.any(result[:, :n_components] != 0)

    def test_single_component(self, synthetic_markers_3d, pca_model):
        result = lower_dim_projection(synthetic_markers_3d, pca_model, n_components=1)
        assert np.any(result[:, 0] != 0)
        assert np.all(result[:, 1:] == 0)

    def test_full_rank(self, synthetic_markers_3d, pca_model):
        n_features = pca_model.components_.shape[1]
        result = lower_dim_projection(synthetic_markers_3d, pca_model, n_components=n_features)
        # No zero padding when using all components
        assert np.any(result[:, -1] != 0)


# -- TestLowerDimReconstruction --

class TestLowerDimReconstruction:
    """Tests for lower_dim_reconstruction()."""

    def test_output_shape(self, synthetic_markers_3d, pca_model):
        result = lower_dim_reconstruction(synthetic_markers_3d, pca_model, n_components=4)
        assert result.shape == synthetic_markers_3d.shape

    def test_full_rank_near_exact(self, synthetic_markers_3d, pca_model):
        """Full-rank reconstruction should near-exactly match original."""
        n_features = pca_model.components_.shape[1]
        result = lower_dim_reconstruction(synthetic_markers_3d, pca_model, n_components=n_features)
        np.testing.assert_allclose(result, synthetic_markers_3d, atol=1e-10)

    def test_rmse_decreases_with_more_components(self, synthetic_markers_3d, pca_model):
        rmses = []
        for nc in [1, 2, 4, 8]:
            recon = lower_dim_reconstruction(synthetic_markers_3d, pca_model, n_components=nc)
            rmse = np.sqrt(np.mean((synthetic_markers_3d - recon) ** 2))
            rmses.append(rmse)
        # RMSE should monotonically decrease (or stay same)
        for i in range(len(rmses) - 1):
            assert rmses[i] >= rmses[i + 1] - 1e-10

    def test_single_component_differs_from_original(self, synthetic_markers_3d, pca_model):
        result = lower_dim_reconstruction(synthetic_markers_3d, pca_model, n_components=1)
        assert not np.allclose(result, synthetic_markers_3d)

    def test_output_dtype_float(self, synthetic_markers_3d, pca_model):
        result = lower_dim_reconstruction(synthetic_markers_3d, pca_model, n_components=4)
        assert result.dtype in [np.float64, np.float32]


# -- TestCalculateReconstructionErrors --

class TestCalculateReconstructionErrors:
    """Tests for calculate_reconstruction_errors()."""

    def test_output_shapes(self, synthetic_markers_3d, pca_model):
        recon = lower_dim_reconstruction(synthetic_markers_3d, pca_model, n_components=4)
        per_frame, per_marker = calculate_reconstruction_errors(synthetic_markers_3d, recon)
        assert per_frame.shape == (100,)
        assert per_marker.shape == (100, 4)

    def test_identical_gives_zero(self, synthetic_markers_3d):
        per_frame, per_marker = calculate_reconstruction_errors(
            synthetic_markers_3d, synthetic_markers_3d
        )
        np.testing.assert_allclose(per_frame, 0, atol=1e-15)
        np.testing.assert_allclose(per_marker, 0, atol=1e-15)

    def test_known_displacement(self):
        """Known displacement of 1.0 in one dimension on one marker."""
        original = np.zeros((5, 2, 3))
        displaced = original.copy()
        displaced[:, 0, 0] = 1.0  # Displace marker 0, x-axis by 1.0
        per_frame, per_marker = calculate_reconstruction_errors(original, displaced)
        # Per marker error for marker 0 should be 1.0 (L2 norm of [1,0,0])
        np.testing.assert_allclose(per_marker[:, 0], 1.0)
        # Per marker error for marker 1 should be 0
        np.testing.assert_allclose(per_marker[:, 1], 0.0)

    def test_all_errors_non_negative(self, synthetic_markers_3d, pca_model):
        recon = lower_dim_reconstruction(synthetic_markers_3d, pca_model, n_components=2)
        per_frame, per_marker = calculate_reconstruction_errors(synthetic_markers_3d, recon)
        assert np.all(per_frame >= 0)
        assert np.all(per_marker >= 0)

    def test_per_frame_is_total_norm(self, synthetic_markers_3d, pca_model):
        """Per-frame error is L2 norm of flattened difference."""
        recon = lower_dim_reconstruction(synthetic_markers_3d, pca_model, n_components=4)
        per_frame, _ = calculate_reconstruction_errors(synthetic_markers_3d, recon)
        diff_flat = (synthetic_markers_3d - recon).reshape(100, -1)
        expected = np.linalg.norm(diff_flat, axis=1)
        np.testing.assert_allclose(per_frame, expected)

    def test_returns_tuple(self, synthetic_markers_3d):
        result = calculate_reconstruction_errors(synthetic_markers_3d, synthetic_markers_3d)
        assert isinstance(result, tuple)
        assert len(result) == 2


# -- TestCalculateMarkerThresholds --

class TestCalculateMarkerThresholds:
    """Tests for calculate_marker_thresholds()."""

    def test_output_shape(self, per_marker_errors_fixture):
        result = calculate_marker_thresholds(per_marker_errors_fixture)
        assert result.shape == (4,)

    def test_wing_and_tail_use_different_percentiles(self, per_marker_errors_fixture):
        result_default = calculate_marker_thresholds(per_marker_errors_fixture)
        result_custom = calculate_marker_thresholds(
            per_marker_errors_fixture, wing_percentile=50, tail_percentile=50
        )
        # Lower percentiles should give lower thresholds
        assert np.all(result_custom <= result_default)

    def test_custom_percentiles_produce_different_thresholds(self, per_marker_errors_fixture):
        result_a = calculate_marker_thresholds(
            per_marker_errors_fixture, wing_percentile=90, tail_percentile=90
        )
        result_b = calculate_marker_thresholds(
            per_marker_errors_fixture, wing_percentile=50, tail_percentile=50
        )
        assert not np.allclose(result_a, result_b)

    def test_thresholds_are_positive(self, per_marker_errors_fixture):
        result = calculate_marker_thresholds(per_marker_errors_fixture)
        assert np.all(result > 0)


# -- TestFilterLowErrorFrames --

class TestFilterLowErrorFrames:
    """Tests for filter_low_error_frames()."""

    def test_returns_boolean_mask(self, per_marker_errors_fixture):
        thresholds = np.full(4, 100.0)  # Very high threshold
        mask = filter_low_error_frames(per_marker_errors_fixture, thresholds)
        assert mask.dtype == bool
        assert len(mask) == len(per_marker_errors_fixture)

    def test_all_below_threshold_all_true(self, per_marker_errors_fixture):
        thresholds = np.full(4, 100.0)
        mask = filter_low_error_frames(per_marker_errors_fixture, thresholds)
        assert np.all(mask)

    def test_all_above_threshold_all_false(self, per_marker_errors_fixture):
        thresholds = np.full(4, 0.0)  # Threshold of 0 — nothing passes (errors > 0)
        mask = filter_low_error_frames(per_marker_errors_fixture, thresholds)
        assert not np.any(mask)

    def test_single_marker_exceeds_excludes_frame(self):
        """A single marker exceeding threshold should exclude the frame."""
        errors = np.array([[0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 5.0]])  # Frame 1: marker 3 high
        thresholds = np.array([1.0, 1.0, 1.0, 1.0])
        mask = filter_low_error_frames(errors, thresholds)
        assert mask[0] == True
        assert mask[1] == False

    def test_prints_statistics(self, per_marker_errors_fixture, capsys):
        thresholds = np.full(4, 2.0)
        filter_low_error_frames(per_marker_errors_fixture, thresholds)
        captured = capsys.readouterr()
        assert "Total frames:" in captured.out
        assert "Excluded frames:" in captured.out
        assert "Remaining frames:" in captured.out

    def test_bug_e7_works_with_5_markers(self):
        """BUG E7: Should work with any number of markers, not just 4."""
        errors = np.random.default_rng(42).uniform(0, 1, size=(50, 5))
        thresholds = np.full(5, 0.5)
        mask = filter_low_error_frames(errors, thresholds)
        assert len(mask) == 50

    def test_mismatched_markers_raises(self):
        """Mismatched marker count between errors and thresholds should raise."""
        errors = np.ones((10, 4))
        thresholds = np.ones(5)
        with pytest.raises(ValueError, match="does not match"):
            filter_low_error_frames(errors, thresholds)


# -- TestClusteringAnalysis --

def _run_clustering_analysis_mocked(data, **kwargs):
    """Run clustering_analysis with matplotlib mocked via sys.modules."""
    import sys
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

    # The function does `from matplotlib import pyplot as plt`
    # so we need matplotlib.pyplot in sys.modules
    mock_matplotlib = MagicMock()
    mock_matplotlib.pyplot = mock_plt
    saved = {
        'matplotlib': sys.modules.get('matplotlib'),
        'matplotlib.pyplot': sys.modules.get('matplotlib.pyplot'),
    }
    sys.modules['matplotlib'] = mock_matplotlib
    sys.modules['matplotlib.pyplot'] = mock_plt
    try:
        result = clustering_analysis(data, **kwargs)
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val
    return result, mock_plt


class TestClusteringAnalysis:
    """Tests for clustering_analysis()."""

    @pytest.fixture
    def clustering_data(self):
        """3D data suitable for clustering_analysis (needs reshaping)."""
        rng = np.random.default_rng(42)
        return rng.normal(size=(200, 4, 3))

    def test_returns_correct_length(self, clustering_data):
        (inertias, silhouettes), _ = _run_clustering_analysis_mocked(
            clustering_data, cluster_range=[2, 3, 4], random_state=42
        )
        assert len(inertias) == 3
        assert len(silhouettes) == 3

    def test_silhouettes_in_valid_range(self, clustering_data):
        (_, silhouettes), _ = _run_clustering_analysis_mocked(
            clustering_data, cluster_range=[2, 3], random_state=42
        )
        for s in silhouettes:
            assert -1 <= s <= 1

    def test_reproducible_with_same_random_state(self, clustering_data):
        (i1, s1), _ = _run_clustering_analysis_mocked(
            clustering_data.copy(), cluster_range=[2, 3], random_state=42
        )
        (i2, s2), _ = _run_clustering_analysis_mocked(
            clustering_data.copy(), cluster_range=[2, 3], random_state=42
        )
        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_array_equal(s1, s2)

    def test_prints_stats(self, clustering_data, capsys):
        _run_clustering_analysis_mocked(
            clustering_data, cluster_range=[2], random_state=42
        )
        captured = capsys.readouterr()
        assert "Clusters:" in captured.out
        assert "Inertia:" in captured.out
        assert "Silhouette:" in captured.out

    def test_calls_matplotlib_show(self, clustering_data):
        _, mock_plt = _run_clustering_analysis_mocked(
            clustering_data, cluster_range=[2], random_state=42
        )
        mock_plt.tight_layout.assert_called_once()
        mock_plt.show.assert_called_once()


# -- TestKmeansClustering --

class TestKmeansClustering:
    """Tests for kmeans_clustering()."""

    @pytest.fixture
    def cluster_data_3d(self):
        rng = np.random.default_rng(42)
        return rng.normal(size=(200, 4, 3))

    def test_centers_shape(self, cluster_data_3d):
        centers, labels = kmeans_clustering(cluster_data_3d, n_clusters=5, random_state=42)
        assert centers.shape == (5, 4, 3)

    def test_labels_shape(self, cluster_data_3d):
        centers, labels = kmeans_clustering(cluster_data_3d, n_clusters=5, random_state=42)
        assert labels.shape == (200,)

    def test_labels_in_valid_range(self, cluster_data_3d):
        _, labels = kmeans_clustering(cluster_data_3d, n_clusters=5, random_state=42)
        assert np.all(labels >= 0)
        assert np.all(labels < 5)

    def test_deterministic(self, cluster_data_3d):
        c1, l1 = kmeans_clustering(cluster_data_3d, n_clusters=3, random_state=42)
        c2, l2 = kmeans_clustering(cluster_data_3d, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(c1, c2)

    def test_different_seeds_may_differ(self, cluster_data_3d):
        _, l1 = kmeans_clustering(cluster_data_3d, n_clusters=3, random_state=1)
        _, l2 = kmeans_clustering(cluster_data_3d, n_clusters=3, random_state=99)
        # Not guaranteed to differ, but labels or centers should likely differ
        # Just check they both have valid structure
        assert l1.shape == l2.shape


# -- TestAnalyseClusters --

class TestAnalyseClusters:
    """Tests for analyse_clusters()."""

    def test_sizes_sum_to_total(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        sizes, _ = analyse_clusters(labels, n_clusters=3)
        assert np.sum(sizes) == 6

    def test_dict_keys_cover_all_clusters(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        _, frame_indices = analyse_clusters(labels, n_clusters=3)
        assert set(frame_indices.keys()) == {0, 1, 2}

    def test_frame_indices_correct(self):
        labels = np.array([0, 1, 0, 1, 2])
        _, frame_indices = analyse_clusters(labels, n_clusters=3)
        np.testing.assert_array_equal(frame_indices[0], [0, 2])
        np.testing.assert_array_equal(frame_indices[1], [1, 3])
        np.testing.assert_array_equal(frame_indices[2], [4])

    def test_prints_statistics(self, capsys):
        labels = np.array([0, 0, 0, 1, 1, 2])
        analyse_clusters(labels, n_clusters=3)
        captured = capsys.readouterr()
        assert "Number of clusters: 3" in captured.out
        assert "Average cluster size:" in captured.out
        assert "Largest cluster:" in captured.out
        assert "Smallest cluster:" in captured.out


# -- TestGenerateKnockOutRepresentations --

class TestGenerateKnockOutRepresentations:
    """Tests for generate_knock_out_representations()."""

    @pytest.fixture
    def cluster_centers(self):
        """3 cluster centers, 4 markers, 3 dims."""
        rng = np.random.default_rng(42)
        return rng.normal(size=(3, 4, 3))

    def test_correct_nested_structure(self, cluster_centers):
        result = generate_knock_out_representations(cluster_centers, [0, 1, 2, 3])
        assert len(result) == 3  # 3 clusters
        assert len(result[0]) == 4  # 4 markers knocked out

    def test_knocked_out_markers_are_nan(self, cluster_centers):
        result = generate_knock_out_representations(cluster_centers, [0, 2])
        # First cluster, first knock-out (marker 0)
        assert np.all(np.isnan(result[0][0][0, :]))
        # First cluster, second knock-out (marker 2)
        assert np.all(np.isnan(result[0][1][2, :]))

    def test_other_markers_unchanged(self, cluster_centers):
        result = generate_knock_out_representations(cluster_centers, [1])
        # Marker 0 should be unchanged in cluster 0, knock-out 0 (marker 1 knocked out)
        np.testing.assert_array_equal(result[0][0][0, :], cluster_centers[0, 0, :])
        np.testing.assert_array_equal(result[0][0][2, :], cluster_centers[0, 2, :])
        np.testing.assert_array_equal(result[0][0][3, :], cluster_centers[0, 3, :])

    def test_does_not_modify_input(self, cluster_centers):
        original = cluster_centers.copy()
        generate_knock_out_representations(cluster_centers, [0, 1, 2, 3])
        np.testing.assert_array_equal(cluster_centers, original)


# -- TestLabellingIntegration --

class TestLabellingIntegration:
    """Integration tests using real test data."""

    def test_full_pipeline_reconstruction_to_filter(self, real_pca_results):
        """Full pipeline: reconstruction → errors → thresholds → filter mask."""
        markers, model = real_pca_results
        # Reconstruct with 4 components
        reconstructed = lower_dim_reconstruction(markers, model, n_components=4)
        assert reconstructed.shape == markers.shape

        # Calculate errors
        per_frame, per_marker = calculate_reconstruction_errors(markers, reconstructed)
        assert per_frame.shape == (markers.shape[0],)
        assert per_marker.shape == (markers.shape[0], markers.shape[1])

        # Calculate thresholds
        thresholds = calculate_marker_thresholds(per_marker)
        assert thresholds.shape == (markers.shape[1],)

        # Filter
        mask = filter_low_error_frames(per_marker, thresholds)
        assert mask.dtype == bool
        assert len(mask) == markers.shape[0]
        # With 99th percentile thresholds, most frames should pass
        assert np.sum(mask) > markers.shape[0] * 0.9

    def test_kmeans_on_real_data(self, real_pca_results):
        """KMeans clustering on real reconstructed data."""
        markers, model = real_pca_results
        reconstructed = lower_dim_reconstruction(markers, model, n_components=4)
        centers, labels = kmeans_clustering(reconstructed, n_clusters=3, random_state=42)
        assert centers.shape == (3, markers.shape[1], 3)
        assert labels.shape == (markers.shape[0],)
        assert set(np.unique(labels)).issubset({0, 1, 2})
