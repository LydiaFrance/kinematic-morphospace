"""
test_clustering.py

Tests for the clustering module: get_cluster_labels(), reorder_cluster_labels(),
restrict_cluster_labels(), and get_cluster_counts().
Covers S13 (Continuum Evidence / Clustering).
"""

import logging
import pytest
import numpy as np
import pandas as pd

from kinematic_morphospace.clustering import (
    get_cluster_labels,
    reorder_cluster_labels,
    restrict_cluster_labels,
    get_cluster_counts,
)

from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


# -- Fixtures --

@pytest.fixture
def synthetic_cluster_data():
    """200 points in 3 well-separated Gaussian blobs (3D)."""
    rng = np.random.default_rng(42)
    centers = np.array([[0, 0, 0], [20, 20, 20], [40, 40, 40]])
    points = []
    for center in centers:
        blob = rng.normal(loc=center, scale=0.5, size=(67, 3))
        points.append(blob)
    # 67*3 = 201, trim to 200
    data = np.vstack(points)[:200]
    return data


@pytest.fixture
def clustered_result(synthetic_cluster_data):
    """Pre-computed get_cluster_labels() output for 3 clusters."""
    labels, centers = get_cluster_labels(synthetic_cluster_data, n_clusters=3, random_state=42)
    return labels, centers, synthetic_cluster_data


@pytest.fixture
def scores_df_with_clusters():
    """DataFrame with known cluster distribution (50/30/20 split)."""
    labels = np.array([0]*50 + [1]*30 + [2]*20)
    return pd.DataFrame({'cluster': labels})


# -- TestGetClusterLabels --

class TestGetClusterLabels:
    """Tests for get_cluster_labels()."""

    def test_returns_tuple(self, synthetic_cluster_data):
        result = get_cluster_labels(synthetic_cluster_data, n_clusters=3, random_state=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_labels_shape(self, synthetic_cluster_data):
        labels, centers = get_cluster_labels(synthetic_cluster_data, n_clusters=3, random_state=42)
        assert labels.shape == (synthetic_cluster_data.shape[0],)

    def test_centers_shape(self, synthetic_cluster_data):
        labels, centers = get_cluster_labels(synthetic_cluster_data, n_clusters=3, random_state=42)
        assert centers.shape == (3, synthetic_cluster_data.shape[1])

    def test_labels_in_valid_range(self, synthetic_cluster_data):
        labels, _ = get_cluster_labels(synthetic_cluster_data, n_clusters=3, random_state=42)
        assert np.all(labels >= 0)
        assert np.all(labels < 3)

    def test_deterministic_with_same_random_state(self, synthetic_cluster_data):
        labels1, centers1 = get_cluster_labels(synthetic_cluster_data, n_clusters=3, random_state=42)
        labels2, centers2 = get_cluster_labels(synthetic_cluster_data, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_equal(centers1, centers2)

    def test_well_separated_blobs_recover_clusters(self, synthetic_cluster_data):
        """Well-separated blobs should be perfectly separated into 3 clusters."""
        labels, _ = get_cluster_labels(synthetic_cluster_data, n_clusters=3, random_state=42)
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 3
        # Each blob should have a consistent label
        # First ~67 points are blob 0, next ~67 are blob 1, last ~66 are blob 2
        for start, end in [(0, 67), (67, 134), (134, 200)]:
            blob_labels = labels[start:end]
            assert len(np.unique(blob_labels)) == 1, (
                f"Points {start}-{end} should all have same label"
            )

    def test_default_n_clusters_is_8(self):
        """Validate BUG E1 fix: default n_clusters should be 8."""
        import inspect
        sig = inspect.signature(get_cluster_labels)
        assert sig.parameters['n_clusters'].default == 8


# -- TestReorderClusterLabels --

class TestReorderClusterLabels:
    """Tests for reorder_cluster_labels()."""

    def test_centroids_reordered_by_distance_from_origin(self):
        """Centroids should be reordered by ascending distance from origin."""
        # Centroid 0 is far, centroid 1 is close, centroid 2 is mid
        centroids = np.array([[10, 10, 10], [1, 1, 1], [5, 5, 5]])
        labels = np.array([0, 0, 1, 1, 2, 2])
        new_labels, new_centroids = reorder_cluster_labels(labels, centroids)
        # After reorder: [1,1,1] < [5,5,5] < [10,10,10]
        distances = np.linalg.norm(new_centroids, axis=1)
        assert np.all(np.diff(distances) >= 0), "Centroids not sorted by distance from origin"

    def test_label_membership_preserved(self):
        """Each point should stay with its centroid after relabelling."""
        centroids = np.array([[10, 10, 10], [1, 1, 1], [5, 5, 5]])
        labels = np.array([0, 0, 1, 1, 2, 2])
        new_labels, new_centroids = reorder_cluster_labels(labels, centroids)
        # Points originally labelled 1 (closest centroid [1,1,1]) should now be label 0
        # Points originally labelled 2 (mid centroid [5,5,5]) should now be label 1
        # Points originally labelled 0 (far centroid [10,10,10]) should now be label 2
        assert np.all(new_labels[:2] == 2)  # originally label 0 → farthest → label 2
        assert np.all(new_labels[2:4] == 0)  # originally label 1 → closest → label 0
        assert np.all(new_labels[4:6] == 1)  # originally label 2 → mid → label 1

    def test_already_ordered_is_noop(self):
        """Centroids already in ascending distance order should not change."""
        centroids = np.array([[1, 0, 0], [3, 0, 0], [5, 0, 0]])
        labels = np.array([0, 1, 2])
        new_labels, new_centroids = reorder_cluster_labels(labels, centroids)
        np.testing.assert_array_equal(labels, new_labels)
        np.testing.assert_array_equal(centroids, new_centroids)

    def test_output_types(self):
        centroids = np.array([[5, 5], [1, 1]])
        labels = np.array([0, 1, 0, 1])
        new_labels, new_centroids = reorder_cluster_labels(labels, centroids)
        assert isinstance(new_labels, np.ndarray)
        assert isinstance(new_centroids, np.ndarray)

    def test_all_labels_present(self):
        """All original cluster indices should still be present after reorder."""
        centroids = np.array([[10, 0], [1, 0], [5, 0]])
        labels = np.array([0, 1, 2, 0, 1, 2])
        new_labels, _ = reorder_cluster_labels(labels, centroids)
        assert set(new_labels) == {0, 1, 2}


# -- TestRestrictClusterLabels --

class TestRestrictClusterLabels:
    """Tests for restrict_cluster_labels()."""

    def test_noise_label_for_distant_points(self, clustered_result):
        """Distant points should get -1 noise label."""
        labels, centers, data = clustered_result
        restricted, _ = restrict_cluster_labels(data, centers, labels, threshold_val=50)
        assert -1 in restricted

    def test_high_threshold_keeps_almost_all(self, clustered_result):
        """threshold_val=100 keeps nearly all points (strict < excludes max)."""
        labels, centers, data = clustered_result
        restricted, _ = restrict_cluster_labels(data, centers, labels, threshold_val=100)
        # Strict '<' means at most a handful of points at exact max distance get -1
        noise_fraction = np.sum(restricted == -1) / len(restricted)
        assert noise_fraction < 0.02

    def test_low_threshold_removes_most(self, clustered_result):
        """Very low threshold should remove most points."""
        labels, centers, data = clustered_result
        restricted, _ = restrict_cluster_labels(data, centers, labels, threshold_val=1)
        noise_count = np.sum(restricted == -1)
        assert noise_count > len(data) * 0.5

    def test_centroids_reordered(self, clustered_result):
        """Returned centroids should be reordered by distance from origin."""
        labels, centers, data = clustered_result
        _, reordered_centers = restrict_cluster_labels(data, centers, labels, threshold_val=70)
        distances = np.linalg.norm(reordered_centers, axis=1)
        assert np.all(np.diff(distances) >= 0)

    def test_output_shapes(self, clustered_result):
        labels, centers, data = clustered_result
        restricted, new_centers = restrict_cluster_labels(data, centers, labels, threshold_val=70)
        assert restricted.shape == (data.shape[0],)
        assert new_centers.shape == centers.shape


# -- TestGetClusterCounts --

class TestGetClusterCounts:
    """Tests for get_cluster_counts()."""

    def test_percentages_sum_to_100(self, scores_df_with_clusters):
        _, counts = get_cluster_counts(scores_df_with_clusters)
        np.testing.assert_almost_equal(np.sum(counts), 100.0)

    def test_known_distribution(self, scores_df_with_clusters):
        """50/30/20 split should produce 50%/30%/20% counts."""
        unique, counts = get_cluster_counts(scores_df_with_clusters)
        np.testing.assert_array_equal(unique, [0, 1, 2])
        np.testing.assert_almost_equal(counts, [50.0, 30.0, 20.0])

    def test_handles_noise_label(self):
        """Should handle -1 noise labels correctly."""
        df = pd.DataFrame({'cluster': np.array([-1, -1, 0, 0, 0, 1, 1, 1, 1, 1])})
        unique, counts = get_cluster_counts(df)
        assert -1 in unique
        np.testing.assert_almost_equal(np.sum(counts), 100.0)

    def test_returns_unique_and_counts(self, scores_df_with_clusters):
        result = get_cluster_counts(scores_df_with_clusters)
        assert isinstance(result, tuple)
        assert len(result) == 2
        unique, counts = result
        assert isinstance(unique, np.ndarray)
        assert isinstance(counts, np.ndarray)


# -- TestContinuumEvidence (R11) --

class TestContinuumEvidence:
    """S13: Continuous (uniformly distributed) data should produce weak
    clustering structure compared to well-separated blobs.  This tests
    the scientific conclusion that hawk flight shapes form a continuum."""

    def test_continuous_data_has_weak_silhouette(self):
        """Uniform data clustered into K groups should have a much lower
        silhouette score than well-separated Gaussian blobs."""
        rng = np.random.default_rng(42)

        # Well-separated blobs
        blobs = np.vstack([
            rng.normal(loc=[0, 0, 0], scale=0.5, size=(100, 3)),
            rng.normal(loc=[20, 20, 20], scale=0.5, size=(100, 3)),
            rng.normal(loc=[40, 40, 40], scale=0.5, size=(100, 3)),
        ])
        blob_labels, _ = get_cluster_labels(blobs, n_clusters=3, random_state=0)
        sil_blobs = silhouette_score(blobs, blob_labels)

        # Continuous uniform data (no cluster structure)
        uniform = rng.uniform(low=0, high=40, size=(300, 3))
        uniform_labels, _ = get_cluster_labels(uniform, n_clusters=3, random_state=0)
        sil_uniform = silhouette_score(uniform, uniform_labels)

        assert sil_blobs > sil_uniform + 0.2, (
            f"Blobs silhouette ({sil_blobs:.3f}) should clearly exceed "
            f"uniform silhouette ({sil_uniform:.3f})"
        )
