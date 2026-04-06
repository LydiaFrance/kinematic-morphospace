"""K-means clustering utilities for PCA score spaces.

Provides helpers for fitting clusters, restricting labels by distance
to centroid, reordering clusters by distance from origin, and
summarising cluster counts.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def get_cluster_labels(data, n_clusters=8, random_state=0):
    """Fit K-means and return cluster labels and centroids.

    Args:
        data: Array of shape (n_samples, n_features) to cluster.
        n_clusters: Number of clusters to fit. Defaults to 8.
        random_state: Random seed for reproducibility. Defaults to 0.

    Returns:
        Tuple of (labels, cluster_centers) where labels is an integer
        array of shape (n_samples,) and cluster_centers is an array
        of shape (n_clusters, n_features).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)

    return kmeans.labels_, kmeans.cluster_centers_


def restrict_cluster_labels(data, cluster_centroids, cluster_labels, threshold_val=70):
    """Remove cluster assignments for points far from their centroid.

    Frames beyond the *threshold_val*-th percentile of within-cluster
    distance are relabelled as noise (``-1``), keeping only the most
    representative frames in each cluster.

    Args:
        data: Array of shape (n_samples, n_features) — the same data used
            to fit the clusters.
        cluster_centroids: Array of shape (n_clusters, n_features) containing
            centroid positions.
        cluster_labels: Integer array of shape (n_samples,) with cluster
            assignments.
        threshold_val: Percentile of within-cluster distance used as the
            cut-off; points beyond this are set to -1. Defaults to 70.

    Returns:
        Tuple of (restricted_labels, reordered_centroids) where
        restricted_labels replaces distant-point assignments with -1
        and reordered_centroids are sorted by distance from the origin.
    """
    cluster_labels, cluster_centroids = reorder_cluster_labels(cluster_labels, cluster_centroids)

    # Calculate the distance between each data point and the cluster centroids
    distances = cdist(data, cluster_centroids, 'euclidean')
    distance_to_centroid = distances[np.arange(len(distances)), cluster_labels]

    # Get the 70th percentile of the distances
    threshold = np.percentile(distance_to_centroid, threshold_val)

    # Restrict the cluster labels based on the threshold, setting noise to -1
    restricted_labels = np.where(distance_to_centroid < threshold, cluster_labels, -1)

    return restricted_labels, cluster_centroids


def reorder_cluster_labels(cluster_labels, cluster_centroids):
    """Relabel clusters in ascending order of centroid distance from the origin.

    Clusters are renumbered so that cluster 0 is closest to the origin
    (the mean shape) and higher-numbered clusters are progressively
    further away in the PCA score space.

    Args:
        cluster_labels: Integer array of shape (n_samples,) with current
            cluster assignments.
        cluster_centroids: Array of shape (n_clusters, n_features) containing
            centroid positions.

    Returns:
        Tuple of (relabelled_clusters, sorted_centroids) where both are
        reordered by ascending distance from the origin.
    """
    # Relabel the clusters based on the total distance from the origin [0,0,0]
    cluster_order = np.argsort(np.linalg.norm(cluster_centroids, axis=1))
    relabelled_clusters = np.zeros_like(cluster_labels)
    for i, cluster in enumerate(cluster_order):
        relabelled_clusters[cluster_labels == cluster] = i

    return relabelled_clusters, cluster_centroids[cluster_order]


def get_cluster_counts(scores_df):
    """Return unique cluster labels and their percentage share of all frames.

    Args:
        scores_df: DataFrame containing a ``cluster`` column with integer
            cluster assignments (``-1`` for noise frames).

    Returns:
        Tuple of (unique_labels, percentages) where unique_labels is a
        sorted array of cluster IDs and percentages is the corresponding
        fraction of total frames expressed as a percentage.
    """
    unique, counts = np.unique(scores_df['cluster'], return_counts=True)
    counts = counts / np.sum(counts) * 100
    return unique, counts
