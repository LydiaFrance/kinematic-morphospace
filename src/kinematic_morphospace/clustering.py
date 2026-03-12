import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def get_cluster_labels(data, n_clusters=8, random_state=0):
    """
    Get cluster labels for data using KMeans clustering.
    
    Parameters
    ----------
    data : np.ndarray
        Data to cluster.
    n_clusters : int, optional
        Number of clusters to use, by default 8.
    random_state : int, optional
        Random seed for reproducibility, by default 0.
    
    Returns
    -------
    np.ndarray
        Cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)

    return kmeans.labels_, kmeans.cluster_centers_

def restrict_cluster_labels(data, cluster_centroids, cluster_labels, threshold_val=70):
    """
    Restrict cluster labels based on the distance between the data and the cluster centers.
    
    Parameters
    ----------
    data : np.ndarray
        Data to cluster.
    cluster_centroids : np.ndarray
        Cluster centre positions.
    cluster_labels : np.ndarray
        Cluster labels.
    threshold_val : int, optional
        Percentile threshold for distance to cluster centre, by default 70.
    
    Returns
    -------
    np.ndarray
        Restricted cluster labels.
    """

    cluster_labels, cluster_centroids = reorder_cluster_labels(cluster_labels,cluster_centroids)

    # Calculate the distance between each data point and the cluster centroids
    distances = cdist(data, cluster_centroids, 'euclidean')
    distance_to_centroid = distances[np.arange(len(distances)), cluster_labels]

    # Get the 70th percentile of the distances
    threshold = np.percentile(distance_to_centroid, threshold_val)

    # Restrict the cluster labels based on the threshold, setting noise to -1
    restricted_labels = np.where(distance_to_centroid < threshold, cluster_labels, -1)
    
    return restricted_labels, cluster_centroids

def reorder_cluster_labels(cluster_labels,cluster_centroids):

    # Relabel the clusters based on the total distance from the origin [0,0,0]
    cluster_order = np.argsort(np.linalg.norm(cluster_centroids, axis=1))
    relabelled_clusters = np.zeros_like(cluster_labels)
    for i, cluster in enumerate(cluster_order):
        relabelled_clusters[cluster_labels == cluster] = i



    return relabelled_clusters, cluster_centroids[cluster_order]

def get_cluster_counts(scores_df):
    """Return unique cluster labels and their percentage counts."""
    unique, counts = np.unique(scores_df['cluster'], return_counts=True)
    counts = counts / np.sum(counts) * 100
    return unique, counts