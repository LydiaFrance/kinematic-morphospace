import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

from ..data_filtering import filter_by
from ..pca_reconstruct import reconstruct


def plot_explained(explained_ratio,
                    ax=None, colour_before=12, annotate=True, ci=None):

    """Plot cumulative explained variance ratio as a coloured bar chart.

    Parameters
    ----------
    explained_ratio : numpy.ndarray
        Explained variance ratio per component.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. A new figure is created when *None*.
    colour_before : int, optional
        Number of leading components to colour individually.
    annotate : bool, optional
        Whether to annotate cumulative-variance thresholds on the plot.
    ci : array-like, optional
        Confidence-interval values (currently unused, reserved for
        future bootstrap intervals).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The parent figure.
    ax : matplotlib.axes.Axes
        The axes containing the bar chart.
    """
    fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.7, 3.5))
        fig.set_constrained_layout(False)

    # bar_colors = ['#CEEEA4', '#89E0B9', '#51B3D4', '#4579AA', '#BC96C9', '#917AC2', '#5A488B']

    bar_colors = ['#B5E675', '#6ED8A9', '#51B3D4',
              '#4579AA', '#F19EBA', '#BC96C9',
              '#917AC2', '#BE607F', '#624E8B',
              '#E6E6E6', '#E6E6E6', '#E6E6E6']

    if colour_before == 0:
        bar_colour = "#51B3D4"
    else:
        bar_colour = "#E6E6E6"

    barlist = plt.bar(range(0,len(explained_ratio)), np.cumsum(explained_ratio),
        color = bar_colour, alpha = 0.8, width = 0.6, edgecolor='None',zorder = 2)

    for i in range(colour_before):
        barlist[i].set_color(bar_colors[i])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', size=0)

    position = ax.get_position()

    # Position the annotation on the right axis.
    # These show where cumulative explained variance is
    # greater than 95%, 97%, and 98%.
    # (Hardcoded values)
    if annotate:
        ax_right = ax.twiny()
        ax_right.spines["bottom"].set_position(("axes", 1.05))
        ax_right.spines["bottom"].set_linewidth(1.5)
        ax_right.xaxis.set_ticks_position("bottom")
        ax_right.xaxis.set_tick_params(width=1.5, length=6)
        ax_right.spines["bottom"].set_visible(True)
        ax_right.set_xticks([-1, -0.33, 0.17, 1])
        ax_right.set_xticklabels(['', '', '', ''])
        ax_right.set_xlim(-1, 1)

        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)

        ax.annotate('>95%', xy=(0.22, 0.94), xycoords='figure fraction')
        ax.annotate('>97%', xy=(0.44, 0.94), xycoords='figure fraction')
        ax.annotate('>98%', xy=(0.68, 0.94), xycoords='figure fraction')

    ax.set_xlabel("Component Number")
    ax.set_ylabel("Cumulative Explained variance ratio")

    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0,1.05,0.1))
    # ax.set_xlim(-0.5,11.5)

    ax.set_xlim(-0.5, len(explained_ratio)-0.5)
    ax.set_xticks(range(0,len(explained_ratio)))
    ax.set_xticklabels(range(1,len(explained_ratio)+1), fontsize=6)
    ax.grid(True, alpha=0.3)

    fig = fig if fig is not None else ax.get_figure()

    return fig, ax


def table_cumulative_variance_ratios(unilateral_data, unilateral_frame_info_df,
                                     principal_components, pca_mean=None):
    """Print cumulative explained variance ratios per hawk, year, and obstacle.

    Iterates over all hawk/year/obstacle combinations, projects each
    subset onto the first nine principal components, and prints the
    cumulative explained variance ratio for each condition.

    Parameters
    ----------
    unilateral_data : numpy.ndarray
        Marker data, shape ``(n_frames, n_markers)``.
    unilateral_frame_info_df : pandas.DataFrame
        Per-frame metadata used for filtering (must contain hawk name,
        year, obstacle, and seqID columns).
    principal_components : numpy.ndarray
        PCA component matrix, shape ``(n_components, n_markers)``.
    pca_mean : numpy.ndarray, optional
        PCA training-set mean, shape ``(n_markers,)``.  When provided
        the data is centred before projection so that variance ratios
        are consistent with the fitted PCA model.

    Returns
    -------
    None
        Results are printed to stdout.
    """

    cumulative_explained_variance_ratios = {}

    if pca_mean is None:
        pca_mean = 0  # No centering when mean not provided

    for hawk in ["Toothless", "Drogon", "Rhaegal", "Ruby", "Charmander"]:
        for year in [2017, 2020]:
            for obs in [0, 1]:
                filter = filter_by(unilateral_frame_info_df, year=year, hawkname=hawk, obstacle=obs)

                if filter.sum() == 0:
                    continue

                subset_data = unilateral_data[filter]
                subset_data = subset_data.reshape(-1, 12)

                # Centre data under the fitted PCA mean before projecting
                subset_centred = subset_data - pca_mean
                subset_projected = np.dot(subset_centred, principal_components[:9,:].T)

                total_variance = np.sum(np.var(subset_data, axis=0))

                explained_variance = np.var(subset_projected, axis=0)

                explained_variance_ratio = explained_variance / total_variance

                # Cumulative explained variance ratio
                cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

                key = f"{hawk} {year} {'obs' if obs == 1 else ''}"

                cumulative_explained_variance_ratios[key] = cumulative_explained_variance_ratio


                num_seqs = np.unique(unilateral_frame_info_df.loc[filter, 'seqID']).shape[0]
                num_frames = np.sum(filter)

                if obs == 0:
                    print(f"{hawk} {year} \t \t  {cumulative_explained_variance_ratio}")
                else:
                    print(f"{hawk} {year} obs \t \t {cumulative_explained_variance_ratio}")


def calculate_cumulative_variance_ratios(data, frame_info_df, principal_components, n_components=9):
    """Calculate cumulative explained variance ratios per condition.

    Projects each hawk/year/obstacle subset onto the principal
    components and returns the cumulative variance ratio for every
    condition that contains data.

    Parameters
    ----------
    data : numpy.ndarray
        Marker data, shape ``(n_frames, n_markers)``.
    frame_info_df : pandas.DataFrame
        Per-frame metadata used for filtering.
    principal_components : numpy.ndarray
        PCA component matrix, shape ``(n_components, n_markers)``.
    n_components : int, optional
        Number of components to project onto (default 9).

    Returns
    -------
    dict
        Mapping of ``"hawk_year_condition"`` strings to
        ``numpy.ndarray`` cumulative variance ratios.
    """
    cumulative_explained_variance_ratios = {}


    for hawk in ["Toothless", "Drogon", "Rhaegal", "Ruby", "Charmander"]:
        for year in [2017, 2020]:
            for obs in [0, 1]:
                filter = filter_by(frame_info_df, year=year, hawkname=hawk, obstacle=obs)

                if filter.sum() == 0:
                    continue

                subset_data = data[filter].reshape(-1, 12)
                subset_projected = np.dot(subset_data, principal_components[:n_components,:].T)

                total_variance = np.sum(np.var(subset_data, axis=0))
                explained_variance = np.var(subset_projected, axis=0)
                explained_variance_ratio = explained_variance / total_variance
                cumulative_ratio = np.cumsum(explained_variance_ratio)

                # Create key that includes obstacle information
                key = f"{hawk}_{year}_{'obs' if obs == 1 else 'straight'}"
                cumulative_explained_variance_ratios[key] = cumulative_ratio

    return cumulative_explained_variance_ratios

def plot_cumulative_variance_ratios(cumulative_explained_variance_ratios):
    """Plot cumulative explained variance ratios for all conditions.

    Draws one line per hawk/year/condition, colour-coded by hawk and
    styled by obstacle presence.

    Parameters
    ----------
    cumulative_explained_variance_ratios : dict
        Mapping of ``"hawk_year_condition"`` keys to
        cumulative variance ratio arrays, as returned by
        :func:`calculate_cumulative_variance_ratios`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the line plot.
    """

    plt.figure(figsize=(10, 6))

    # Define colour scheme — matches standard hawk palette used project-wide
    individual_colors = {
        "Toothless":  "#66C2A5",
        "Drogon":     "#FC8D62",
        "Rhaegal":    "#8DA0CB",
        "Ruby":       "#E78AC3",
        "Charmander": "#A6D854",
    }

    # Define line styles
    conditions_styles = {
        'straight': '-',
        'obs': '--'
    }

    for key, cum_exp_var_ratio in cumulative_explained_variance_ratios.items():
        hawk, year, condition = key.split('_')
        plt.plot(range(1, len(cum_exp_var_ratio) + 1),
                cum_exp_var_ratio,
                marker='o',
                label=f"{hawk} {year} {condition}",
                color=individual_colors[hawk],
                linestyle=conditions_styles[condition])

    plt.title('Cumulative Explained Variance Ratio by Principal Components')
    plt.xlabel('Morphing Shape Mode')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.05)
    plt.xlim(0.5, 9.5)
    plt.xticks(range(1, 10))
    plt.grid(True)
    plt.tight_layout()

    return plt.gcf()


def plot_explained_comparison(real_explained, shuffled_explained, ax=None):
    """Compare cumulative variance of real data against a shuffled control.

    Overlays the shuffled-data cumulative variance as a black line on
    top of the coloured real-data bar chart, highlighting the
    additional structure captured by the PCA.

    Parameters
    ----------
    real_explained : numpy.ndarray
        Explained variance ratio per component for the real data.
    shuffled_explained : numpy.ndarray
        Explained variance ratio per component for the shuffled control.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. A new figure is created when *None*.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The parent figure.
    ax : matplotlib.axes.Axes
        The axes containing the comparison plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.7, 3.5))
        fig.set_constrained_layout(False)

    # Plot cumulative explained variance for real data
    real_cum = np.cumsum(real_explained)
    shuffled_cum = np.cumsum(shuffled_explained)

    # Plot real with colours
    bar_colors = ['#B5E675', '#6ED8A9', '#51B3D4',
                  '#4579AA', '#F19EBA', '#BC96C9',
                  '#917AC2', '#BE607F', '#624E8B',
                  '#E6E6E6', '#E6E6E6', '#E6E6E6']

    barlist = ax.bar(range(len(real_cum)), real_cum, color="#E6E6E6", width=0.6, edgecolor='None', zorder=2)
    for i in range(min(len(bar_colors), len(barlist))):
        barlist[i].set_color(bar_colors[i])

    # Plot shuffled as line on top
    ax.plot(range(len(shuffled_cum)), shuffled_cum, marker='o', linestyle='-', color='black', label='Shuffled control', zorder=3)

    # Formatting
    ax.set_xlabel("Component Number")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_ylim(0, 1.01)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_xlim(-0.5, len(real_cum) - 0.5)
    ax.set_xticks(range(len(real_cum)))
    ax.set_xticklabels(range(1, len(real_cum) + 1), fontsize=6)
    ax.grid(True, alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', size=0)

    # Legend
    ax.legend(loc='lower right', fontsize=9)

    fig = fig if fig is not None else ax.get_figure()

    return fig, ax


def plot_hist_similar_shapes(principal_components, scores, marker_data,
                              pc_indices=None, threshold=0.025):
    """Plot histograms of similar shapes at each PC score.

    Counts how many observed frames fall within a threshold RMS distance
    of reconstructed shapes at each PC score. Also produces 2-D
    adaptive-resolution similar-shape maps for PC1 vs PC2.

    Parameters
    ----------
    principal_components : numpy.ndarray
        PCA component matrix.
    scores : numpy.ndarray
        Score matrix, shape ``(n_frames, n_components)``.
    marker_data : numpy.ndarray
        Original marker data, shape ``(n_frames, n_markers, 3)``.
    pc_indices : list of int, optional
        PC indices to analyse. Defaults to ``[0, 1]``.
    threshold : float, optional
        RMS distance threshold for counting similar shapes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : numpy.ndarray of matplotlib.axes.Axes
        Array of subplot axes.
    """
    def compare_shapes(shapes1, shapes2):
        """Compute RMS difference between two sets of shapes."""
        diff = shapes1[:, None, :, :] - shapes2[None, :, :, :]
        distances = np.sqrt(np.mean(diff**2, axis=(2, 3)))
        return distances

    if pc_indices is None:
        pc_indices = [0, 1]

    n_bins = 50
    n_components = principal_components.shape[0]

    colourList = ['#B5E675', '#6ED8A9', '#51B3D4',
                  '#4579AA', '#BC96C9', '#917AC2',
                  '#5A488B', '#888888', '#888888',
                  '#888888', '#888888', '#888888']

    n_markers = marker_data.shape[1]
    mu = marker_data.mean(axis=0).reshape(1, n_markers, 3)

    n_rows = len(pc_indices) + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, n_rows * 2.5), sharey='row')

    if n_rows == 1:
        axes = np.array([axes])

    # 1D histograms
    for i, pc_index in enumerate(pc_indices):
        pc_scores = scores[:, pc_index]
        score_min, score_max = np.min(pc_scores), np.max(pc_scores)
        score_bins = np.linspace(score_min, score_max, n_bins)

        test_score_vectors = np.zeros((n_bins, n_components))
        test_score_vectors[:, pc_index] = score_bins

        reconstructed_shapes = reconstruct(test_score_vectors, principal_components,
                                            mu, components_list=[pc_index])
        distances = compare_shapes(reconstructed_shapes, marker_data)
        frame_counts = np.sum(distances < threshold, axis=1)

        ax_left = axes[i, 0]
        ax_left.bar(score_bins, frame_counts, width=(score_bins[1] - score_bins[0]) * 0.9,
                    color=colourList[pc_index], align='center')
        ax_left.set_title(f'PC{pc_index + 1}: Similar Shapes')
        ax_left.set_xlabel(f'PC{pc_index + 1} Score')
        ax_left.set_ylabel('Frame Count')

        ax_right = axes[i, 1]
        ax_right.hist(pc_scores, bins=n_bins, color=colourList[pc_index], edgecolor='k')
        ax_right.set_title(f'PC{pc_index + 1}: Score Frequency')
        ax_right.set_xlabel(f'PC{pc_index + 1} Score')
        ax_right.set_ylabel('Frequency')

    # 2D histogram with adaptive resolution
    if len(pc_indices) >= 2 and 0 in pc_indices and 1 in pc_indices:
        custom_cmap = LinearSegmentedColormap.from_list('white_to_green', ['white', '#4FB42D'])

        pc1_scores = scores[:, 0]
        pc2_scores = scores[:, 1]
        pc1_min, pc1_max = np.min(pc1_scores), np.max(pc1_scores)
        pc2_min, pc2_max = np.min(pc2_scores), np.max(pc2_scores)

        # Coarse grid to identify regions of interest
        coarse_bins = 20
        pc1_coarse_bins = np.linspace(pc1_min, pc1_max, coarse_bins)
        pc2_coarse_bins = np.linspace(pc2_min, pc2_max, coarse_bins)

        pc1_coarse_mesh, pc2_coarse_mesh = np.meshgrid(pc1_coarse_bins, pc2_coarse_bins)
        pc1_coarse_flat = pc1_coarse_mesh.flatten()
        pc2_coarse_flat = pc2_coarse_mesh.flatten()

        test_score_vectors_coarse = np.zeros((len(pc1_coarse_flat), n_components))
        test_score_vectors_coarse[:, 0] = pc1_coarse_flat
        test_score_vectors_coarse[:, 1] = pc2_coarse_flat

        reconstructed_shapes_coarse = reconstruct(test_score_vectors_coarse,
                                                    principal_components, mu,
                                                    components_list=[0, 1])
        distances_coarse = compare_shapes(reconstructed_shapes_coarse, marker_data)
        frame_counts_coarse = np.sum(distances_coarse < threshold, axis=1)
        frame_counts_coarse_mesh = frame_counts_coarse.reshape(pc2_coarse_mesh.shape)

        count_threshold = np.percentile(
            frame_counts_coarse_mesh[frame_counts_coarse_mesh > 0], 75)
        high_count_regions = frame_counts_coarse_mesh >= count_threshold

        # Fine grid for score frequency
        fine_bins = 50
        hist_2d, pc1_fine_edges, pc2_fine_edges = np.histogram2d(
            pc1_scores, pc2_scores,
            bins=[fine_bins, fine_bins],
            range=[[pc1_min, pc1_max], [pc2_min, pc2_max]]
        )

        pc1_fine_centers = (pc1_fine_edges[:-1] + pc1_fine_edges[1:]) / 2
        pc2_fine_centers = (pc2_fine_edges[:-1] + pc2_fine_edges[1:]) / 2
        pc1_fine_mesh, pc2_fine_mesh = np.meshgrid(pc1_fine_centers, pc2_fine_centers)

        fine_shape_counts = np.zeros(pc1_fine_mesh.shape)

        # High-resolution computation in regions of interest
        coarse_i_indices = np.clip(
            np.floor((pc2_fine_mesh - pc2_min) / (pc2_max - pc2_min) * (coarse_bins - 1)).astype(int),
            0, coarse_bins - 1)
        coarse_j_indices = np.clip(
            np.floor((pc1_fine_mesh - pc1_min) / (pc1_max - pc1_min) * (coarse_bins - 1)).astype(int),
            0, coarse_bins - 1)

        high_res_mask = high_count_regions[coarse_i_indices, coarse_j_indices]
        high_res_indices = np.where(high_res_mask)
        pc1_high_res = pc1_fine_mesh[high_res_indices]
        pc2_high_res = pc2_fine_mesh[high_res_indices]

        n_high_res = len(pc1_high_res)
        n_total = fine_shape_counts.size
        print(f"Computing high resolution for {n_high_res} points out of "
              f"{n_total} ({n_high_res / n_total * 100:.1f}%)")

        batch_size = 100
        for batch_start in range(0, n_high_res, batch_size):
            batch_end = min(batch_start + batch_size, n_high_res)
            batch_indices = np.arange(batch_start, batch_end)

            batch_score_vectors = np.zeros((len(batch_indices), n_components))
            batch_score_vectors[:, 0] = pc1_high_res[batch_indices]
            batch_score_vectors[:, 1] = pc2_high_res[batch_indices]

            batch_shapes = reconstruct(batch_score_vectors, principal_components,
                                        mu, components_list=[0, 1])
            batch_distances = compare_shapes(batch_shapes, marker_data)
            batch_counts = np.sum(batch_distances < threshold, axis=1)

            i_indices = high_res_indices[0][batch_indices]
            j_indices = high_res_indices[1][batch_indices]
            fine_shape_counts[i_indices, j_indices] = batch_counts

        # Interpolate low-count regions from coarse grid
        low_res_mask = ~high_res_mask
        if np.any(low_res_mask):
            from scipy.interpolate import RegularGridInterpolator
            interp_func = RegularGridInterpolator(
                (pc2_coarse_bins, pc1_coarse_bins),
                frame_counts_coarse_mesh,
                bounds_error=False,
                fill_value=0
            )
            low_res_indices = np.where(low_res_mask)
            points_to_interp = np.column_stack([
                pc2_fine_mesh[low_res_indices], pc1_fine_mesh[low_res_indices]])
            fine_shape_counts[low_res_indices] = interp_func(points_to_interp)

        # Create the 2D plots
        ax_left_2d = axes[-1, 0]
        im_left = ax_left_2d.pcolormesh(pc1_fine_mesh, pc2_fine_mesh, fine_shape_counts,
                                         cmap=custom_cmap, shading='auto', vmin=0, vmax=5000)
        fig.colorbar(im_left, ax=ax_left_2d, label='Count')
        ax_left_2d.set_title('PC1 vs PC2: Similar Shapes')
        ax_left_2d.set_xlabel('PC1 Score')
        ax_left_2d.set_ylabel('PC2 Score')
        ax_left_2d.set_ylim(-0.7, 0.3)

        ax_right_2d = axes[-1, 1]
        im_right = ax_right_2d.pcolormesh(
            (pc1_fine_edges[:-1] + pc1_fine_edges[1:]) / 2,
            (pc2_fine_edges[:-1] + pc2_fine_edges[1:]) / 2,
            hist_2d.T,
            cmap=custom_cmap, shading='auto', vmin=0, vmax=300
        )
        fig.colorbar(im_right, ax=ax_right_2d, label='Count')
        ax_right_2d.set_title('PC1 vs PC2: Score Frequency')
        ax_right_2d.set_xlabel('PC1 Score')
        ax_right_2d.set_ylabel('PC2 Score')

    plt.tight_layout()
    return fig, axes
