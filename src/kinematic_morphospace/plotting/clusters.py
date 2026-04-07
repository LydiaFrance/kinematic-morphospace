"""Cluster visualisation functions.

Combines plotting functions previously in labelling.py and clustering.py.
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ..data_filtering import filter_by

# --- From clustering.py ---

def get_cluster_colours(labels, n_clusters=8, noise_colour='blanchedalmond'):
    """Map cluster labels to RGBA colours using the Set3 colourmap.

    Args:
        labels: Cluster label per frame; may include -1 for noise points.
        n_clusters: Total number of valid clusters (excluding noise). Defaults to 8.
        noise_colour: Named colour for noise points (label -1). Defaults to
            'blanchedalmond'.

    Returns:
        Tuple of (colours_by_cluster, colour_map) where colours_by_cluster is a
        list of RGBA tuples aligned with labels, and colour_map is a dict mapping
        each cluster index to its RGBA colour.
    """
    set3 = plt.cm.get_cmap('Set3')
    cluster_colours = [set3(i) for i in range(set3.N)]

    # Swap the 5th colour with the 3rd colour
    cluster_colours[4], cluster_colours[1] = cluster_colours[1], cluster_colours[4]

    colour_map = {i: cluster_colours[i] for i in range(n_clusters)}
    colour_map[-1] = mcolors.to_rgba(noise_colour)

    colours_by_cluster = [colour_map[label] for label in labels]

    return colours_by_cluster, colour_map


def plot_clusters(
    selected_scores,
    cluster_centroids,
    cluster_colours,
    PC_pair=None,
    ax=None,
):
    """Scatter plot of PCA scores coloured by cluster assignment.

    Each point represents one frame; colour indicates the cluster it belongs
    to. Cluster indices are annotated near their centroids.

    Args:
        selected_scores: Score matrix of shape (n_frames, n_PCs).
        cluster_centroids: Centroid coordinates of shape (n_clusters, n_PCs).
        cluster_colours: List of RGBA colours aligned with selected_scores rows,
            as returned by get_cluster_colours().
        PC_pair: Two-element list of PC column indices to plot on the x and y
            axes respectively. Defaults to [0, 1].
        ax: Matplotlib Axes object to draw on; if None, a new figure is created.
    """
    if PC_pair is None:
        PC_pair = [0, 1]

    standalone = ax is None
    if standalone:
        _fig, ax = plt.subplots(figsize=(4, 6))

    PC_1 = PC_pair[1]
    PC_2 = PC_pair[0]

    ax.scatter(selected_scores[:, PC_1], selected_scores[:, PC_2],
               marker='o', c=cluster_colours, alpha=0.6, s=1, edgecolors='none')

    for i in range(cluster_centroids.shape[0]):
        if i != -1:
            ax.text(cluster_centroids[i, PC_pair[1]] - 0.01,
                    cluster_centroids[i, PC_pair[0]] - 0.01,
                    str(i), fontsize=12, color='black')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel(f'PC{PC_1 + 1}')
    ax.set_ylabel(f'PC{PC_2 + 1}')

    if standalone:
        plt.show()


def get_cluster_counts(scores_df, all_labels=None):
    """Return unique cluster labels and their percentage of total frames.

    Args:
        scores_df: DataFrame with a ``cluster`` column.
        all_labels: If provided, counts are returned for every label in this
            sequence (inserting zeros for clusters absent from scores_df).
            This prevents shape mismatches when comparing conditions.

    Returns:
        Tuple of (labels, counts) where labels is an array of unique cluster
        indices and counts is the percentage of frames in each cluster.
    """
    unique, counts = np.unique(scores_df['cluster'], return_counts=True)
    counts = counts / np.sum(counts) * 100

    if all_labels is not None:
        full_counts = np.zeros(len(all_labels))
        for i, label in enumerate(all_labels):
            idx = np.where(unique == label)[0]
            if len(idx) > 0:
                full_counts[i] = counts[idx[0]]
        return np.array(all_labels), full_counts

    return unique, counts


def plot_cluster_counts(unique, counts, colour_list, title=None, fig=None, ax=None):
    """Bar chart showing the percentage of frames assigned to each cluster.

    Args:
        unique: Array of cluster label values.
        counts: Percentage of total frames in each cluster, aligned with unique.
        colour_list: Dict mapping cluster index to RGBA colour, as returned by
            get_cluster_colours().
        title: Optional subplot title text.
        fig: Existing Figure to draw into; if None, a new figure is created.
        ax: Existing Axes to draw into; if None, a new axes is created.
    """
    standalone = fig is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    colour_array = np.array(list(colour_list.values()))
    ax.bar(unique, counts, color=colour_array)

    n_clusters = len(unique)
    ax.set_xticks(range(n_clusters))

    if title is not None:
        ax.set_title(title, fontsize=10, y=0.8)

    if standalone:
        plt.show()


def _ax_settings_cluster_diffs(ax, y_lim, y_tick):
    """Apply standard formatting to cluster difference subplots."""
    ax.set_ylim(-y_lim, y_lim)
    ax.set_yticks([-y_tick, 0, y_tick])
    ax.set_yticklabels([-y_tick, 0, y_tick], fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    return ax


def plot_cluster_diffs(scores_df, colour_list):
    """Plot cluster count differences between control and treatment conditions.

    Creates a 4x4 grid of bar charts. Each row is one hawk (Drogon,
    Toothless, Ruby, Charmander); columns show the absolute control counts
    followed by differences for weight, obstacle, and experience conditions.
    Noise frames (cluster == -1) are excluded.

    Args:
        scores_df: DataFrame containing PC scores, cluster labels, and metadata.
        colour_list: Dict mapping cluster index to RGBA colour.
    """
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    no_noise = scores_df['cluster'] != -1
    no_noise_scores_df = scores_df[no_noise]

    # Get all non-noise cluster labels for consistent counting
    all_labels = sorted(set(no_noise_scores_df['cluster'].unique()) - {-1})

    counter = 0
    for hawk in ["Drogon", "Toothless", "Ruby", "Charmander"]:
        selection = filter_by(no_noise_scores_df, perchDist=9, obstacle=0,
                              IMU=0, hawkname=hawk, year=2020)
        unique_labels, counts = get_cluster_counts(
            no_noise_scores_df[selection], all_labels=all_labels)
        plot_cluster_counts(unique_labels, counts, colour_list, f"{hawk}",
                            fig=fig, ax=axes[counter])
        axes[counter].set_ylim(0, 60)
        axes[counter].spines['top'].set_visible(False)
        axes[counter].spines['bottom'].set_visible(False)
        axes[counter].spines['right'].set_visible(False)
        counter += 1

        selection = filter_by(no_noise_scores_df, perchDist=9, obstacle=0,
                              IMU=1, hawkname=hawk, year=2020)
        _, counts_compare = get_cluster_counts(
            no_noise_scores_df[selection], all_labels=all_labels)
        diff_counts = counts_compare - counts
        plot_cluster_counts(unique_labels, diff_counts, colour_list, "with weight",
                            fig=fig, ax=axes[counter])
        axes[counter].set_ylim(-25, 25)
        axes[counter].axhline(0, color='black', linewidth=0.8, linestyle="--")
        _ax_settings_cluster_diffs(axes[counter], 25, 25)
        counter += 1

        selection = filter_by(no_noise_scores_df, perchDist=9, obstacle=1,
                              IMU=0, hawkname=hawk, year=2020)
        _, counts_compare = get_cluster_counts(
            no_noise_scores_df[selection], all_labels=all_labels)
        diff_counts = counts_compare - counts
        plot_cluster_counts(unique_labels, diff_counts, colour_list, "with obstacle",
                            fig=fig, ax=axes[counter])
        axes[counter].axhline(0, color='black', linewidth=0.8, linestyle="--")
        _ax_settings_cluster_diffs(axes[counter], 25, 25)
        counter += 1

        if hawk.startswith("Charmander"):
            axes[counter].axis('off')
            continue

        selection = filter_by(no_noise_scores_df, perchDist=9, obstacle=0,
                              IMU=0, hawkname=hawk, year=2017)
        _, counts_compare = get_cluster_counts(
            no_noise_scores_df[selection], all_labels=all_labels)
        diff_counts = counts - counts_compare
        plot_cluster_counts(unique_labels, diff_counts, colour_list, "with experience",
                            fig=fig, ax=axes[counter])
        axes[counter].axhline(0, color='black', linewidth=0.8, linestyle="--")
        _ax_settings_cluster_diffs(axes[counter], 40, 40)
        counter += 1

    plt.show()


def plot_cluster_experience_diffs(scores_df, colour_list):
    """Plot cluster count differences between naive and experienced flights.

    Creates a 3x2 grid of bar charts: left column shows absolute cluster
    counts for the experienced year; right column shows the count difference
    (experienced minus naive). Only hawks with both 2017 and 2020 data are
    included.

    Args:
        scores_df: DataFrame containing PC scores, cluster labels, and metadata.
        colour_list: Dict mapping cluster index to RGBA colour.
    """
    fig, axes = plt.subplots(3, 2, figsize=(6, 6))
    axes = axes.flatten()

    no_noise = scores_df['cluster'] != -1
    no_noise_scores_df = scores_df[no_noise]

    # Get all non-noise cluster labels for consistent counting
    all_labels = sorted(set(no_noise_scores_df['cluster'].unique()) - {-1})

    counter = 0
    for hawk in ["Drogon", "Toothless", "Ruby"]:
        selection = filter_by(no_noise_scores_df, perchDist=9, obstacle=0,
                              IMU=0, hawkname=hawk, year=2020)
        unique_labels, counts = get_cluster_counts(
            no_noise_scores_df[selection], all_labels=all_labels)
        plot_cluster_counts(unique_labels, counts, colour_list, f"{hawk}",
                            fig=fig, ax=axes[counter])
        axes[counter].set_ylim(0, 50)
        axes[counter].spines['top'].set_visible(False)
        axes[counter].spines['bottom'].set_visible(False)
        counter += 1

        selection = filter_by(no_noise_scores_df, perchDist=9, obstacle=0,
                              IMU=0, hawkname=hawk, year=2017)
        _, counts_compare = get_cluster_counts(
            no_noise_scores_df[selection], all_labels=all_labels)
        diff_counts = counts - counts_compare
        plot_cluster_counts(unique_labels, diff_counts, colour_list, "with experience",
                            fig=fig, ax=axes[counter])
        axes[counter].set_ylim(-25, 25)
        axes[counter].axhline(0, color='black', linewidth=0.8, linestyle="--")
        axes[counter].spines['top'].set_visible(False)
        axes[counter].spines['bottom'].set_visible(False)
        axes[counter].spines['right'].set_visible(False)
        axes[counter].set_xticks([])
        axes[counter].set_yticks([-25, 0, 25])
        counter += 1

    plt.show()


# --- From labelling.py ---

def plot_reconstruction_errors(errors, percentile=98):
    """Plot a histogram of per-frame reconstruction errors with a percentile threshold.

    Used to identify outlier frames where the PCA reconstruction is poor,
    which typically indicates labelling errors or unusual wing postures.

    Args:
        errors: Per-frame reconstruction error values.
        percentile: Percentile used to compute the threshold line. Frames above
            this threshold are considered outliers. Defaults to 98.

    Returns:
        Tuple of (threshold, num_bad_frames) where threshold is the error value
        at the given percentile and num_bad_frames is the count of frames above it.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Number of Frames')
    plt.title('Histogram of Reconstruction Errors for Frames')

    threshold = np.percentile(errors, percentile)
    plt.axvline(threshold, color='red', linestyle='--',
                label=f'{percentile}th Percentile ({threshold:.2f})')
    plt.legend()
    plt.show()

    num_bad_frames = np.sum(errors > threshold)
    print(
        f"Number of frames above the {percentile}th percentile threshold: "
        f"{num_bad_frames}"
    )
    print(
        f"Error value at the {percentile}th percentile threshold: "
        f"{threshold}"
    )
    return threshold, num_bad_frames


def plot_marker_errors_with_thresholds(
    data,
    per_marker_errors,
    per_marker_thresholds,
    marker_labels,
    view_labels,
):
    """Scatter plot of per-marker positions with high reconstruction error.

    Creates a 4x3 grid (one row per marker, three columns for XY, XZ, YZ
    views). Frames whose reconstruction error exceeds the per-marker
    threshold are drawn in red at larger size; well-reconstructed frames are
    drawn in black.

    Args:
        data: Marker position array of shape (n_frames, n_markers, 3).
        per_marker_errors: Per-frame, per-marker reconstruction errors of shape
            (n_frames, n_markers).
        per_marker_thresholds: Error threshold for each marker, length n_markers.
        marker_labels: Names for each marker (used as subplot titles).
        view_labels: Labels for each 2-D projection (e.g. ['XY', 'XZ', 'YZ']).
    """
    _fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    base_size, error_size = 0.5, 2

    for ii in range(4):
        threshold = per_marker_thresholds[ii]
        for jj, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)]):
            colors = np.where(
                per_marker_errors[:, ii] > threshold, 'red', 'black'
            )
            sizes = np.where(
                per_marker_errors[:, ii] > threshold, error_size, base_size
            )
            alphas = np.where(
                per_marker_errors[:, ii] > threshold, 0.5, 0.1
            )

            axes[ii, jj].scatter(data[:, ii, x], data[:, ii, y],
                                  c=colors, s=sizes, alpha=alphas, edgecolor="none")
            axes[ii, jj].set_title(
                f"{marker_labels[ii]} - {view_labels[jj]} (Threshold: {threshold:.3f})")
            axes[ii, jj].set_xlabel(['X', 'X', 'Y'][jj])
            axes[ii, jj].set_ylabel(['Y', 'Z', 'Z'][jj])
            axes[ii, jj].set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_cluster_size_distribution(cluster_sizes):
    """Plot a histogram of HDBSCAN cluster sizes."""
    plt.figure(figsize=(10, 5))
    plt.hist(cluster_sizes, bins=50)
    plt.xlabel('Cluster Size (number of frames)')
    plt.ylabel('Number of Clusters')
    plt.title('Distribution of Cluster Sizes')
    plt.show()
