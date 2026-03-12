"""Dimensionality reduction, reconstruction, and clustering for marker labelling.

Provides PCA-based lower-dimensional projection, reconstruction-error
analysis, and mini-batch K-means clustering utilities used to assign
shape labels to motion-capture frames.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

def lower_dim_reconstruction(markers, pca_model, n_components=4):

    """
    Reconstruct data from the padded lower-dimensional PCA projection.
    """

    n_markers = markers.shape[1]

    projected_markers = lower_dim_projection(markers, pca_model, n_components)

    reconstructed_data = pca_model.inverse_transform(projected_markers)
    reconstructed_data = reconstructed_data.reshape(-1, n_markers, 3)  # Reshape to original marker dimensions
    return reconstructed_data

def calculate_reconstruction_errors(markers, reconstructed_markers):
    """
    Calculate the reconstruction error per frame and per marker.
    """

    n_markers = markers.shape[1]
    n_dims = markers.shape[2]

    # shape [n_frames]
    markers_frames = markers.reshape(-1, n_markers*n_dims)
    reconstructed_frames = reconstructed_markers.reshape(-1, n_markers*n_dims)
    reconstruction_errors = np.linalg.norm(markers_frames - reconstructed_frames, 
                                           axis=1) 
    # shape [n_frames, n_markers]
    per_marker_errors = np.linalg.norm(markers - reconstructed_markers, 
                                       axis=2)  # Per marker error
    
    return reconstruction_errors, per_marker_errors


def calculate_marker_thresholds(per_marker_errors, wing_percentile=99, tail_percentile=99.7):
    """
    Calculate threshold for each marker based on specified percentiles.
    """
    wing_thresholds = np.percentile(per_marker_errors[:, :3], wing_percentile, axis=0)  # Wing markers
    tail_threshold = np.percentile(per_marker_errors[:, 3], tail_percentile)  # Tail marker
    per_marker_thresholds = np.concatenate((wing_thresholds, [tail_threshold]))
    return per_marker_thresholds

def filter_low_error_frames(per_marker_errors, per_marker_thresholds):
    """
    Create a mask to filter frames based on marker-specific error thresholds.
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
    """Evaluate clustering over a range of cluster counts.

    Computes inertia and silhouette scores for each value in
    *cluster_range* and plots elbow and silhouette curves.

    Parameters
    ----------
    data : np.ndarray
        Data to cluster (already projected), with shape
        ``(n_samples, n_markers, n_dimensions)``.
    cluster_range : iterable of int
        Cluster counts to evaluate.
    sample_size : int, optional
        Number of samples used for the silhouette score (default 10 000).
    is_log_scale : bool, optional
        Whether to use a log scale for the cluster-count axis.
    title : str, optional
        Title for the plots.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    inertias : list of float
        Inertia value for each cluster count.
    silhouettes : list of float
        Silhouette score for each cluster count.
    """
    inertias = []
    silhouettes = []
    rng = np.random.default_rng(random_state)

    # Flatten the data to 2d
    data = data.reshape(-1, data.shape[1]*data.shape[2])

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

        # Sample subset for silhouette score calculation
        sample_indices = rng.choice(len(data), min(sample_size, len(data)), replace=False)
        silhouette = silhouette_score(data[sample_indices], kmeans.predict(data[sample_indices]))
        silhouettes.append(silhouette)
        
        print(f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.0f}, Silhouette: {silhouette:.3f}")
    
    # Plot results
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
    """Run mini-batch K-means clustering and return reshaped centres.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape ``(n_samples, n_markers, n_dimensions)``.
    n_clusters : int
        Number of clusters to form.
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    cluster_centres : np.ndarray
        Cluster centres of shape ``(n_clusters, n_markers, n_dimensions)``.
    labels : np.ndarray
        Cluster label for each sample.
    """
    # Flatten data for clustering (n_samples, markers * dimensions)
    n_samples, markers, dimensions = data.shape
    flattened_data = data.reshape(n_samples, markers * dimensions)
    
    # Perform KMeans clustering
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        n_init='auto',
        max_iter=100
    )
    kmeans.fit(flattened_data)
    
    # Reshape cluster centres to original dimensions (n_clusters, markers, dimensions)
    cluster_centers = kmeans.cluster_centers_.reshape(n_clusters, markers, dimensions)
    
    return cluster_centers, kmeans.labels_

def analyse_clusters(kmeans_labels, n_clusters):
    """Calculate and print summary statistics for cluster assignments.

    Parameters
    ----------
    kmeans_labels : np.ndarray
        Cluster label assigned to each frame.
    n_clusters : int
        Total number of clusters.

    Returns
    -------
    cluster_sizes : np.ndarray
        Number of frames in each cluster.
    cluster_frame_indices : dict
        Mapping from cluster index to an array of frame indices.
    """
    # Get cluster sizes
    cluster_sizes = np.bincount(kmeans_labels)
    
    # Get frame indices for each cluster
    cluster_frame_indices = {i: np.where(kmeans_labels == i)[0] for i in range(n_clusters)}

    # Print cluster statistics
    print(f"Number of clusters: {n_clusters}")
    print(f"Average cluster size: {np.mean(cluster_sizes):.1f} frames")
    print(f"Median cluster size: {np.median(cluster_sizes):.1f} frames")
    print(f"Largest cluster: {np.max(cluster_sizes)} frames")
    print(f"Smallest cluster: {np.min(cluster_sizes)} frames")
    
    return cluster_sizes, cluster_frame_indices


def generate_knock_out_representations(cluster_centers, missing_marker_indices):
    """
    Generate knock-out representations for clusters by setting specified markers to NaN.
    """
    knock_out_representations = []
    for cluster in cluster_centers:
        knock_out_versions = []
        for missing_idx in missing_marker_indices:
            modified_cluster = np.copy(cluster)
            modified_cluster[missing_idx, :] = np.nan  # Knock-out marker
            knock_out_versions.append(modified_cluster)
        knock_out_representations.append(knock_out_versions)
    return knock_out_representations

# ---------- Helpers ----------


def lower_dim_projection(markers, pca_model, n_components=4):
    """Project markers into a lower-dimensional PCA space.

    Transforms *markers* using *pca_model*, retains only the first
    *n_components* scores, and zero-pads the remaining dimensions so
    the result can be passed to ``pca_model.inverse_transform()``.

    Parameters
    ----------
    markers : np.ndarray
        Marker array of shape ``(n_frames, n_markers, 3)``.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.
    n_components : int, optional
        Number of leading components to retain (default 4).

    Returns
    -------
    np.ndarray
        Zero-padded score array of shape ``(n_frames, n_features)``.
    """

    n_markers = markers.shape[1]
    n_dims = markers.shape[2]

    projected_data = pca_model.transform(markers.reshape(-1, n_markers*n_dims))[:, :n_components]
    padded_projected = np.zeros((projected_data.shape[0], pca_model.components_.shape[1]))
    padded_projected[:, :n_components] = projected_data
    return padded_projected



