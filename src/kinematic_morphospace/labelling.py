"""Dimensionality reduction, reconstruction, and clustering for marker labelling.

Provides PCA-based lower-dimensional projection, reconstruction-error
analysis, and mini-batch K-means clustering utilities used to assign
shape labels to motion-capture frames.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def lower_dim_reconstruction(markers, pca_model, n_components=4):
    """Reconstruct marker positions from a truncated lower-dimensional PCA projection.

    Projects markers into the leading ``n_components`` PCA dimensions, then
    reconstructs back to the original marker space. The difference between
    the reconstruction and the original is the reconstruction error.

    Args:
        markers: Marker array of shape ``(n_frames, n_markers, 3)``.
        pca_model: Fitted sklearn PCA model.
        n_components: Number of leading PCA components to retain during
            reconstruction. Defaults to 4.

    Returns:
        Reconstructed marker array of shape ``(n_frames, n_markers, 3)``.
    """
    n_markers = markers.shape[1]

    projected_markers = lower_dim_projection(markers, pca_model, n_components)

    reconstructed_data = pca_model.inverse_transform(projected_markers)
    reconstructed_data = reconstructed_data.reshape(-1, n_markers, 3)
    return reconstructed_data


def calculate_reconstruction_errors(markers, reconstructed_markers):
    """Calculate the per-frame and per-marker reconstruction errors.

    Computes L2 (Euclidean) norm errors at both the frame level (all
    markers combined) and the individual marker level, which are used
    downstream to identify outlier frames and noisy markers.

    Args:
        markers: Original marker array of shape ``(n_frames, n_markers, 3)``.
        reconstructed_markers: Reconstructed marker array of the same shape.

    Returns:
        Tuple of (reconstruction_errors, per_marker_errors) where
        reconstruction_errors is a 1-D array of shape ``(n_frames,)``
        and per_marker_errors is an array of shape ``(n_frames, n_markers)``.
    """
    n_markers = markers.shape[1]
    n_dims = markers.shape[2]

    # shape [n_frames]
    markers_frames = markers.reshape(-1, n_markers * n_dims)
    reconstructed_frames = reconstructed_markers.reshape(-1, n_markers * n_dims)
    reconstruction_errors = np.linalg.norm(markers_frames - reconstructed_frames, axis=1)

    # shape [n_frames, n_markers]
    per_marker_errors = np.linalg.norm(markers - reconstructed_markers, axis=2)

    return reconstruction_errors, per_marker_errors


def calculate_marker_thresholds(per_marker_errors, wing_percentile=99, tail_percentile=99.7):
    """Calculate per-marker error thresholds based on specified percentiles.

    Wing markers (first three) and the tail marker use different percentile
    thresholds because tail tracking is noisier and warrants a more
    permissive cutoff.

    Args:
        per_marker_errors: Array of shape ``(n_frames, n_markers)`` containing
            per-marker reconstruction errors.
        wing_percentile: Percentile threshold applied to the three wing markers.
            Defaults to 99.
        tail_percentile: Percentile threshold applied to the tail marker (index 3).
            Defaults to 99.7.

    Returns:
        Array of shape ``(n_markers,)`` with the per-marker error thresholds.
    """
    wing_thresholds = np.percentile(per_marker_errors[:, :3], wing_percentile, axis=0)
    tail_threshold = np.percentile(per_marker_errors[:, 3], tail_percentile)
    per_marker_thresholds = np.concatenate((wing_thresholds, [tail_threshold]))
    return per_marker_thresholds


def filter_low_error_frames(per_marker_errors, per_marker_thresholds):
    """Create a boolean mask that keeps only frames with low reconstruction error.

    A frame is retained only if every marker's error is below its
    corresponding threshold. Frames that exceed any threshold are excluded
    as likely mis-tracked or anomalous.

    Args:
        per_marker_errors: Array of shape ``(n_frames, n_markers)`` with
            per-marker reconstruction errors.
        per_marker_thresholds: Array of shape ``(n_markers,)`` with per-marker
            error cutoffs, e.g. from :func:`calculate_marker_thresholds`.

    Returns:
        Boolean mask of shape ``(n_frames,)`` where True indicates a frame
        that passes all thresholds.

    Raises:
        ValueError: If the number of markers in ``per_marker_errors`` does not
            match the length of ``per_marker_thresholds``.
    """
    if per_marker_errors.shape[1] != len(per_marker_thresholds):
        raise ValueError(
            f"Number of markers in errors ({per_marker_errors.shape[1]}) does not match "
            f"number of thresholds ({len(per_marker_thresholds)})"
        )
    low_error_mask = np.ones(len(per_marker_errors), dtype=bool)
    for marker_idx in range(per_marker_errors.shape[1]):
        low_error_mask &= (per_marker_errors[:, marker_idx] <= per_marker_thresholds[marker_idx])

    total_frames = len(per_marker_errors)
    low_error_frames = np.sum(low_error_mask)
    excluded_frames = total_frames - low_error_frames

    print(f"Total frames: {total_frames}")
    print(f"Excluded frames: {excluded_frames} ({(excluded_frames / total_frames) * 100:.1f}%)")
    print(f"Remaining frames: {low_error_frames} ({(low_error_frames / total_frames) * 100:.1f}%)")

    return low_error_mask


# ---------- Clustering ----------


def clustering_analysis(data, cluster_range, sample_size=10000, is_log_scale=True, title="Clustering Analysis", random_state=42):
    """Evaluate clustering quality over a range of cluster counts.

    Computes inertia and silhouette scores for each value in
    ``cluster_range`` and plots elbow and silhouette curves to aid in
    selecting the optimal number of clusters.

    Args:
        data: Data to cluster, with shape ``(n_samples, n_markers, n_dimensions)``.
            It is flattened to 2-D before clustering.
        cluster_range: Iterable of integer cluster counts to evaluate.
        sample_size: Number of frames sampled for the silhouette score
            to keep computation tractable. Defaults to 10000.
        is_log_scale: Whether to use a log scale for the cluster-count axis
            in the plots. Defaults to True.
        title: Title for the elbow and silhouette plots. Defaults to
            ``"Clustering Analysis"``.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (inertias, silhouettes) where each is a list of floats
        with one value per entry in ``cluster_range``.
    """
    inertias = []
    silhouettes = []
    rng = np.random.default_rng(random_state)

    data = data.reshape(-1, data.shape[1] * data.shape[2])

    for n_clusters in cluster_range:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=4096,
            n_init='auto',
            max_iter=100
        )

        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

        sample_indices = rng.choice(len(data), min(sample_size, len(data)), replace=False)
        silhouette = silhouette_score(data[sample_indices], kmeans.predict(data[sample_indices]))
        silhouettes.append(silhouette)

        print(f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.0f}, Silhouette: {silhouette:.3f}")

    from matplotlib import pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if is_log_scale:
        ax1.semilogx(cluster_range, inertias, 'bo-')
    else:
        ax1.plot(cluster_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters' + (' (log scale)' if is_log_scale else ''))
    ax1.set_ylabel('Inertia')
    ax1.set_title(f'Elbow Method ({title})')
    ax1.grid(True)

    if is_log_scale:
        ax2.semilogx(cluster_range, silhouettes, 'ro-')
    else:
        ax2.plot(cluster_range, silhouettes, 'ro-')
    ax2.set_xlabel('Number of Clusters' + (' (log scale)' if is_log_scale else ''))
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title(f'Silhouette Analysis ({title})')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return inertias, silhouettes


def kmeans_clustering(data, n_clusters, random_state=42):
    """Run mini-batch K-means clustering and return reshaped cluster centres.

    Args:
        data: Input array of shape ``(n_samples, n_markers, n_dimensions)``.
        n_clusters: Number of clusters to form.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (cluster_centres, labels) where cluster_centres has shape
        ``(n_clusters, n_markers, n_dimensions)`` and labels is an integer
        array of shape ``(n_samples,)``.
    """
    n_samples, markers, dimensions = data.shape
    flattened_data = data.reshape(n_samples, markers * dimensions)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        n_init='auto',
        max_iter=100
    )
    kmeans.fit(flattened_data)

    cluster_centers = kmeans.cluster_centers_.reshape(n_clusters, markers, dimensions)

    return cluster_centers, kmeans.labels_


def analyse_clusters(kmeans_labels, n_clusters):
    """Calculate and print summary statistics for cluster assignments.

    Args:
        kmeans_labels: Integer array of cluster labels of shape ``(n_frames,)``.
        n_clusters: Total number of clusters used in the fit.

    Returns:
        Tuple of (cluster_sizes, cluster_frame_indices) where cluster_sizes
        is an array of shape ``(n_clusters,)`` with frame counts per cluster,
        and cluster_frame_indices is a dict mapping cluster index to an array
        of frame indices.
    """
    cluster_sizes = np.bincount(kmeans_labels)

    cluster_frame_indices = {i: np.where(kmeans_labels == i)[0] for i in range(n_clusters)}

    print(f"Number of clusters: {n_clusters}")
    print(f"Average cluster size: {np.mean(cluster_sizes):.1f} frames")
    print(f"Median cluster size: {np.median(cluster_sizes):.1f} frames")
    print(f"Largest cluster: {np.max(cluster_sizes)} frames")
    print(f"Smallest cluster: {np.min(cluster_sizes)} frames")

    return cluster_sizes, cluster_frame_indices


def generate_knock_out_representations(cluster_centers, missing_marker_indices):
    """Generate knock-out cluster representations by setting specified markers to NaN.

    For each cluster centre, creates one variant per missing-marker index
    with the specified marker coordinates replaced by NaN. Used to test
    how reconstruction-based labelling degrades when markers are occluded.

    Args:
        cluster_centers: Array of shape ``(n_clusters, n_markers, 3)``
            containing cluster centroid shapes.
        missing_marker_indices: Iterable of marker indices to knock out.

    Returns:
        Nested list of shape ``[n_clusters][n_missing_markers]`` where each
        element is a copy of the cluster centre with the specified marker
        set to NaN.
    """
    knock_out_representations = []
    for cluster in cluster_centers:
        knock_out_versions = []
        for missing_idx in missing_marker_indices:
            modified_cluster = np.copy(cluster)
            modified_cluster[missing_idx, :] = np.nan
            knock_out_versions.append(modified_cluster)
        knock_out_representations.append(knock_out_versions)
    return knock_out_representations


# ---------- Helpers ----------


def lower_dim_projection(markers, pca_model, n_components=4):
    """Project markers into a truncated PCA space, zero-padding the remaining dimensions.

    Transforms ``markers`` using ``pca_model``, retains only the first
    ``n_components`` scores, and zero-pads the remaining dimensions so
    the result can be passed directly to ``pca_model.inverse_transform()``.

    Args:
        markers: Marker array of shape ``(n_frames, n_markers, 3)``.
        pca_model: Fitted sklearn PCA model.
        n_components: Number of leading components to retain. Defaults to 4.

    Returns:
        Zero-padded score array of shape ``(n_frames, n_features)`` where
        only the first ``n_components`` columns are non-zero.
    """
    n_markers = markers.shape[1]
    n_dims = markers.shape[2]

    projected_data = pca_model.transform(markers.reshape(-1, n_markers * n_dims))[:, :n_components]
    padded_projected = np.zeros((projected_data.shape[0], pca_model.components_.shape[1]))
    padded_projected[:, :n_components] = projected_data
    return padded_projected
