"""Explained variance and scree plots for PCA morphospace analysis.

Functions here visualise cumulative explained variance ratios, compare
real data against null (shuffled) controls, and provide per-condition
breakdowns across hawks, years, and flight conditions.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator

from ..data_filtering import filter_by
from ..pca_reconstruct import reconstruct


def plot_explained(
    explained_ratio,
    ax=None,
    colour_before=12,
    annotate=True,
    _ci=None,
):
    """Plot cumulative explained variance ratio as a colour-coded bar chart.

    Each bar represents one additional principal component; bar height is the
    cumulative explained variance up to that component. Leading components are
    colour-coded individually using the standard project palette; remaining
    components are shown in light grey.

    Args:
        explained_ratio: Explained variance ratio per component, length n_comp.
        ax: Matplotlib Axes to draw on; if None, a new figure is created.
        colour_before: Number of leading components to colour individually.
            Defaults to 12 (all components coloured).
        annotate: When True, annotates the x-axis with >95%, >97%, and >98%
            cumulative-variance threshold markers. Defaults to True.
        ci: Reserved for future bootstrap confidence-interval bands; currently
            unused. Defaults to None.

    Returns:
        Tuple of (fig, ax).
    """
    fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.7, 3.5), constrained_layout=False)
    assert ax is not None

    # bar_colors = [
    #     '#CEEEA4', '#89E0B9', '#51B3D4', '#4579AA',
    #     '#BC96C9', '#917AC2', '#5A488B'
    # ]

    bar_colors = ['#B5E675', '#6ED8A9', '#51B3D4',
              '#4579AA', '#F19EBA', '#BC96C9',
              '#917AC2', '#BE607F', '#624E8B',
              '#E6E6E6', '#E6E6E6', '#E6E6E6']

    bar_colour = "#51B3D4" if colour_before == 0 else "#E6E6E6"

    barlist = plt.bar(
        range(len(explained_ratio)),
        np.cumsum(explained_ratio),
        color=bar_colour,
        alpha=0.8,
        width=0.6,
        edgecolor='None',
        zorder=2,
    )

    for i in range(colour_before):
        barlist[i].set_color(bar_colors[i])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', size=0)

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
    ax.set_xticks(range(len(explained_ratio)))
    ax.set_xticklabels([str(i) for i in range(1, len(explained_ratio)+1)], fontsize=6)
    ax.grid(True, alpha=0.3)

    fig = fig if fig is not None else ax.get_figure()

    return fig, ax


def table_cumulative_variance_ratios(
    unilateral_data, unilateral_frame_info_df, principal_components, pca_mean=None
):
    """Print cumulative explained variance ratios per hawk/year/condition.

    Projects each subset of frames onto the first nine principal components and
    prints the cumulative explained variance ratio. Used to verify that the
    pooled PCA explains a consistent proportion of variance across conditions.

    Args:
        unilateral_data: Marker data array of shape (n_frames, n_markers).
        unilateral_frame_info_df: Per-frame metadata DataFrame used for
            filtering; must contain hawk name, year, obstacle, and seqID
            columns.
        principal_components: PCA component matrix of shape
            (n_components, n_markers).
        pca_mean: PCA training-set mean of shape (n_markers,). When provided
            the data is centred before projection so that variance ratios are
            consistent with the fitted PCA model. Defaults to None (no
            centring).
    """
    cumulative_explained_variance_ratios = {}

    if pca_mean is None:
        pca_mean = 0  # No centering when mean not provided

    for hawk in ["Toothless", "Drogon", "Rhaegal", "Ruby", "Charmander"]:
        for year in [2017, 2020]:
            for obs in [0, 1]:
                filter = filter_by(
                    unilateral_frame_info_df,
                    year=year,
                    hawkname=hawk,
                    obstacle=obs,
                )

                if filter.sum() == 0:
                    continue

                subset_data = unilateral_data[filter]
                subset_data = subset_data.reshape(-1, 12)

                # Centre data under the fitted PCA mean before projecting
                subset_centred = subset_data - pca_mean
                subset_projected = np.dot(
                    subset_centred, principal_components[:9, :].T
                )

                total_variance = np.sum(np.var(subset_data, axis=0))

                explained_variance = np.var(subset_projected, axis=0)

                explained_variance_ratio = explained_variance / total_variance

                # Cumulative explained variance ratio
                cumulative_explained_variance_ratio = np.cumsum(
                    explained_variance_ratio
                )

                period = "Period 1" if year == 2017 else "Period 2"
                key = f"{hawk} {period} {'obs' if obs == 1 else ''}"

                cumulative_explained_variance_ratios[key] = (
                    cumulative_explained_variance_ratio
                )


                if obs == 0:
                    print(
                        f"{hawk} {period} \t \t  "
                        f"{cumulative_explained_variance_ratio}"
                    )
                else:
                    print(
                        f"{hawk} {period} obs \t \t "
                        f"{cumulative_explained_variance_ratio}"
                    )


def calculate_cumulative_variance_ratios(
    data, frame_info_df, principal_components, n_components=9
):
    """Calculate cumulative explained variance ratios per condition.

    Projects each condition's frames onto the principal components and returns
    the cumulative variance ratio for every condition that has data. The result
    can be passed directly to plot_cumulative_variance_ratios().

    Args:
        data: Marker data array of shape (n_frames, n_markers).
        frame_info_df: Per-frame metadata DataFrame used for filtering.
        principal_components: PCA component matrix of shape
            (n_components, n_markers).
        n_components: Number of components to project onto. Defaults to 9.

    Returns:
        Dict mapping 'hawk_Period N_condition' strings to cumulative variance
        ratio arrays of length n_components.
    """
    cumulative_explained_variance_ratios = {}


    for hawk in ["Toothless", "Drogon", "Rhaegal", "Ruby", "Charmander"]:
        for year in [2017, 2020]:
            for obs in [0, 1]:
                filter = filter_by(
                    frame_info_df, year=year, hawkname=hawk, obstacle=obs
                )

                if filter.sum() == 0:
                    continue

                subset_data = data[filter].reshape(-1, 12)
                subset_projected = np.dot(
                    subset_data, principal_components[:n_components, :].T
                )

                total_variance = np.sum(np.var(subset_data, axis=0))
                explained_variance = np.var(subset_projected, axis=0)
                explained_variance_ratio = explained_variance / total_variance
                cumulative_ratio = np.cumsum(explained_variance_ratio)

                # Create key that includes obstacle information
                period = "Period 1" if year == 2017 else "Period 2"
                key = (
                    f"{hawk}_{period}_{'obs' if obs == 1 else 'straight'}"
                )
                cumulative_explained_variance_ratios[key] = cumulative_ratio

    return cumulative_explained_variance_ratios

def plot_cumulative_variance_ratios(cumulative_explained_variance_ratios):
    """Plot cumulative explained variance ratios for all combinations.

    Draws one line per condition, colour-coded by hawk and styled by year and
    obstacle presence. This allows rapid visual comparison of how much variance
    each condition's frames are captured by the pooled PCA.

    Args:
        cumulative_explained_variance_ratios: Dict mapping
            'hawk_Period N_condition' key strings to cumulative variance ratio
            arrays, as returned by calculate_cumulative_variance_ratios().

    Returns:
        Figure containing the cumulative variance ratio line plot.
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

    # Period → linestyle, condition → marker
    period_styles = {
        'Period 1': '-',
        'Period 2': '--',
    }
    condition_markers = {
        'straight': 'o',
        'obs': '^',
    }

    for key, cum_exp_var_ratio in (
        cumulative_explained_variance_ratios.items()
    ):
        # Key format: "Hawk_Period N_condition"
        # split from right to preserve "Period N"
        parts = key.rsplit('_', 1)
        hawk_period, condition = parts[0], parts[1]
        hawk, period = hawk_period.split('_', 1)
        label_condition = "obstacle" if condition == "obs" else "straight"
        plt.plot(
            range(1, len(cum_exp_var_ratio) + 1),
            cum_exp_var_ratio,
            marker=condition_markers[condition],
            markersize=4,
            label=f"{hawk} {period} {label_condition}",
            color=individual_colors[hawk],
            linestyle=period_styles.get(period, '-'),
            lw=1.5,
        )

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


def plot_explained_comparison(real_explained, shuffled_explained, ax=None,):
    """Compare cumulative explained variance of real data against a shuffled control.

    Overlays the shuffled-data cumulative variance as a black line on top of the
    colour-coded real-data bar chart. The gap between the real and shuffled curves
    reveals the additional structure captured by the PCA that is not explained
    by random correlations.

    Args:
        real_explained: Explained variance ratio per component for the real data.
        shuffled_explained: Explained variance ratio per component for the
            shuffled (null) control.
        ax: Matplotlib Axes to draw on; if None, a new figure is created.

    Returns:
        Tuple of (fig, ax).
    """
    fig = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.7, 3.5), constrained_layout=False)
    assert ax is not None

    # Plot cumulative explained variance for real data
    real_cum = np.cumsum(real_explained)
    shuffled_cum = np.cumsum(shuffled_explained)

    # Plot real with colours
    bar_colors = ['#B5E675', '#6ED8A9', '#51B3D4',
                  '#4579AA', '#F19EBA', '#BC96C9',
                  '#917AC2', '#BE607F', '#624E8B',
                  '#E6E6E6', '#E6E6E6', '#E6E6E6']

    barlist = ax.bar(
        range(len(real_cum)),
        real_cum,
        color="#E6E6E6",
        width=0.6,
        edgecolor='None',
        zorder=2,
    )
    for i in range(min(len(bar_colors), len(barlist))):
        barlist[i].set_color(bar_colors[i])

    # Plot shuffled as line on top
    ax.plot(
        range(len(shuffled_cum)),
        shuffled_cum,
        marker='o',
        linestyle='-',
        color='black',
        label='Shuffled control',
        zorder=3,
    )

    # Formatting
    ax.set_xlabel("Component Number")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_ylim(0, 1.01)
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_xlim(-0.5, len(real_cum) - 0.5)
    ax.set_xticks(range(len(real_cum)))
    ax.set_xticklabels([str(i) for i in range(1, len(real_cum) + 1)], fontsize=6)
    ax.grid(True, alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', size=0)

    # Legend
    ax.legend(loc='lower right', fontsize=9)

    fig = fig if fig is not None else ax.get_figure()

    return fig, ax


def plot_hist_similar_shapes(
    principal_components, scores, marker_data, pc_indices=None, threshold=0.025
):
    """Plot histograms and 2-D maps of how many observed shapes resemble each PC score.

    For each PC in pc_indices, counts how many observed frames have an RMS
    distance below the threshold from the reconstructed shape at each score
    value. This reveals which regions of the morphospace are densely populated
    versus rarely occupied. Also produces adaptive-resolution 2-D maps for PC1
    vs PC2.

    Args:
        principal_components: PCA component matrix of shape (n_components, n_markers).
        scores: PCA score matrix of shape (n_frames, n_components).
        marker_data: Observed marker data of shape (n_frames, n_markers, 3).
        pc_indices: List of PC indices (0-based) to analyse. Defaults to [0, 1].
        threshold: RMS distance threshold in metres for counting similar shapes.
            Defaults to 0.025.

    Returns:
        Tuple of (fig, axes).
    """
    def compare_shapes(shapes1, shapes2):
        """Compute RMS difference between two sets of shapes."""
        diff = shapes1[:, None, :, :] - shapes2[None, :, :, :]
        return np.sqrt(np.mean(diff**2, axis=(2, 3)))

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

        reconstructed_shapes = reconstruct(
            test_score_vectors,
            principal_components,
            mu,
            components_list=[pc_index],
        )
        distances = compare_shapes(reconstructed_shapes, marker_data)
        frame_counts = np.sum(distances < threshold, axis=1)

        ax_left = axes[i, 0]
        ax_left.hist(
            pc_scores, bins=n_bins, color=colourList[pc_index], edgecolor='k'
        )
        ax_left.set_title(f'PC{pc_index + 1}: Score Frequency')
        ax_left.set_xlabel(f'PC{pc_index + 1} Score')
        ax_left.set_ylabel('Frequency')

        ax_right = axes[i, 1]
        ax_right.bar(
            score_bins,
            frame_counts,
            width=(score_bins[1] - score_bins[0]) * 0.9,
            color=colourList[pc_index],
            align='center',
        )
        ax_right.set_title(f'PC{pc_index + 1}: Similar Shapes')
        ax_right.set_xlabel(f'PC{pc_index + 1} Score')
        ax_right.set_ylabel('Frame Count')

    # 2D histogram with adaptive resolution
    if len(pc_indices) >= 2 and 0 in pc_indices and 1 in pc_indices:
        custom_cmap = LinearSegmentedColormap.from_list(
            'white_to_green', ['white', '#4FB42D']
        )

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

        reconstructed_shapes_coarse = reconstruct(
            test_score_vectors_coarse,
            principal_components,
            mu,
            components_list=[0, 1],
        )
        distances_coarse = compare_shapes(
            reconstructed_shapes_coarse, marker_data
        )
        frame_counts_coarse = np.sum(distances_coarse < threshold, axis=1)
        frame_counts_coarse_mesh = frame_counts_coarse.reshape(
            pc2_coarse_mesh.shape
        )

        count_threshold = np.percentile(
            frame_counts_coarse_mesh[frame_counts_coarse_mesh > 0], 75
        )
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
            np.floor(
                (pc2_fine_mesh - pc2_min)
                / (pc2_max - pc2_min)
                * (coarse_bins - 1)
            ).astype(int),
            0,
            coarse_bins - 1,
        )
        coarse_j_indices = np.clip(
            np.floor(
                (pc1_fine_mesh - pc1_min)
                / (pc1_max - pc1_min)
                * (coarse_bins - 1)
            ).astype(int),
            0,
            coarse_bins - 1,
        )

        high_res_mask = high_count_regions[coarse_i_indices, coarse_j_indices]
        high_res_indices = np.where(high_res_mask)
        pc1_high_res = pc1_fine_mesh[high_res_indices]
        pc2_high_res = pc2_fine_mesh[high_res_indices]

        n_high_res = len(pc1_high_res)
        n_total = fine_shape_counts.size
        print(
            f"Computing high resolution for {n_high_res} points out of "
            f"{n_total} ({n_high_res / n_total * 100:.1f}%)"
        )

        batch_size = 100
        for batch_start in range(0, n_high_res, batch_size):
            batch_end = min(batch_start + batch_size, n_high_res)
            batch_indices = np.arange(batch_start, batch_end)

            batch_score_vectors = np.zeros((len(batch_indices), n_components))
            batch_score_vectors[:, 0] = pc1_high_res[batch_indices]
            batch_score_vectors[:, 1] = pc2_high_res[batch_indices]

            batch_shapes = reconstruct(
                batch_score_vectors,
                principal_components,
                mu,
                components_list=[0, 1],
            )
            batch_distances = compare_shapes(batch_shapes, marker_data)
            batch_counts = np.sum(batch_distances < threshold, axis=1)

            i_indices = high_res_indices[0][batch_indices]
            j_indices = high_res_indices[1][batch_indices]
            fine_shape_counts[i_indices, j_indices] = batch_counts

        # Interpolate low-count regions from coarse grid
        low_res_mask = ~high_res_mask
        if np.any(low_res_mask):
            interp_func = RegularGridInterpolator(
                (pc2_coarse_bins, pc1_coarse_bins),
                frame_counts_coarse_mesh,
                bounds_error=False,
                fill_value=0,
            )
            low_res_indices = np.where(low_res_mask)
            points_to_interp = np.column_stack(
                [
                    pc2_fine_mesh[low_res_indices],
                    pc1_fine_mesh[low_res_indices],
                ]
            )
            fine_shape_counts[low_res_indices] = interp_func(
                points_to_interp
            )

        # Create the 2D plots
        ax_left_2d = axes[-1, 0]
        im_left = ax_left_2d.pcolormesh(
            (pc1_fine_edges[:-1] + pc1_fine_edges[1:]) / 2,
            (pc2_fine_edges[:-1] + pc2_fine_edges[1:]) / 2,
            hist_2d.T,
            cmap=custom_cmap,
            shading='auto',
            vmin=0,
            vmax=300,
        )
        fig.colorbar(im_left, ax=ax_left_2d, label='Count')
        ax_left_2d.set_title('PC1 vs PC2: Score Frequency')
        ax_left_2d.set_xlabel('PC1 Score')
        ax_left_2d.set_ylabel('PC2 Score')

        ax_right_2d = axes[-1, 1]
        im_right = ax_right_2d.pcolormesh(
            pc1_fine_mesh,
            pc2_fine_mesh,
            fine_shape_counts,
            cmap=custom_cmap,
            shading='auto',
            vmin=0,
            vmax=5000,
        )
        fig.colorbar(im_right, ax=ax_right_2d, label='Count')
        ax_right_2d.set_title('PC1 vs PC2: Similar Shapes')
        ax_right_2d.set_xlabel('PC1 Score')
        ax_right_2d.set_ylabel('PC2 Score')
        ax_right_2d.set_ylim(-0.7, 0.3)

    plt.tight_layout()

    # Show the figure if it was created inside this function
    if fig is not None:
        plt.show()
    return fig, axes
