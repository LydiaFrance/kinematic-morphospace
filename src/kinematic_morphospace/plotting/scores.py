"""PC score time-trace and morphospace scatter plots.

Functions here display how PC scores evolve along the approach trajectory (binned
by horizontal distance to the perch) and provide pairwise scatter diagnostics for
exploring the structure of the morphospace.
"""
import numpy as np
from matplotlib import pyplot as plt

from ..pca_scores import get_binned_scores


def plot_score(scores_df, PC_name='PC01', ax=None, alpha=1, linestyle='-',
               **filters):
    """Plot a single PC score trace against horizontal distance to the perch.

    Bins the scores by horizontal distance and draws the mean score as a line
    with ±1 standard deviation shading. Axis limits are determined from the
    flight phase (excluding the perch-grab region within 1 m of the perch).

    Args:
        scores_df: DataFrame containing PC scores and per-frame flight metadata.
        PC_name: Name of the PC column to plot (e.g. 'PC01'). Defaults to 'PC01'.
        ax: Matplotlib Axes object to draw on; if None, a new figure is created.
        alpha: Opacity of the mean line. Defaults to 1.
        linestyle: Line style for the mean trace (e.g. '-', '--'). Defaults to '-'.
        **filters: Keyword arguments forwarded to get_binned_scores() for
            subsetting the data (e.g. hawkname='Drogon', obstacle=0).

    Returns:
        Axes object containing the score trace.
    """
    if ax is None:
        _fig, ax = plt.subplots()

    colour_PC_dict = {
        'PC01': '#B5E675',
        'PC02': '#6ED8A9',
        'PC03': '#51B3D4',
        'PC04': '#4579AA',
        'PC05': '#F19EBA',
        'PC06': '#BC96C9',
        'PC07': '#917AC2',
        'PC08': '#BE607F',
        'PC09': '#624E8B',
        'PC10': '#888888',
        'PC11': '#888888',
        'PC12': '#888888',
    }


    # Bins the scores by horizontal distance to the perch
    # Filters provided by user, e.g. only obstacle flights or by individual bird
    binned_info, mean_scores, stdev_scores, _med_scores = get_binned_scores(
        scores_df, **filters
    )

    score = mean_scores[PC_name]
    stdev = stdev_scores[PC_name]
    horzDist_bins = binned_info['HorzDistance']

    # If any scores are nan, remove that row and the corresponding row in
    # horzDist_bins and stdev
    nan_rows = np.isnan(stdev)
    score = score[~nan_rows]
    stdev = stdev[~nan_rows]
    horzDist_bins = horzDist_bins[~nan_rows]

    # Provide a zero score axis
    ax.axhline(y=0, color='#333333', linestyle=':', linewidth=0.5)

    # Shade +/-1 standard deviation
    ax.fill_between(
        horzDist_bins,
        score - stdev,
        score + stdev,
        color=colour_PC_dict[PC_name],
        alpha=0.4,
        edgecolor='none',
    )

    # Plot the mean per bin
    ax.plot(
        horzDist_bins,
        score,
        color=colour_PC_dict[PC_name],
        linewidth=2,
        alpha=alpha,
        linestyle=linestyle,
    )

    # Set y-axis limits excluding the perch zone (last 1 m) where
    # extreme values from the grab distort the scale
    flight_mask = horzDist_bins < -1.0
    if flight_mask.any():
        y_max = np.max((score + stdev)[flight_mask])
        y_min = np.min((score - stdev)[flight_mask])
    else:
        y_max = np.max(score + stdev)
        y_min = np.min(score - stdev)
    padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - padding, y_max + padding)

    return ax

def plot_score_multi_PCs(scores_df, PC_num_list=None, **filters):
    """Plot binned score traces for multiple morphing shape modes in a grid.

    Creates one panel per PC. When all nine interpretable modes are requested,
    PC8 is re-ordered adjacent to PC5 for thematic grouping. The grid is sized
    dynamically (up to 3 columns, as many rows as needed).

    Args:
        scores_df: DataFrame containing PC scores and per-frame metadata.
        PC_num_list: PC numbers to include; defaults to 1-12.
        **filters: Keyword arguments forwarded to plot_score(). Must include
            perchDist.

    Returns:
        Figure containing the grid of score traces.

    Raises:
        ValueError: If perchDist is not included in filters.
    """
    if PC_num_list is None:
        PC_num_list = range(1, 13)
    if 'perchDist' not in filters:
        msg = "perchDist should be in filters"
        raise ValueError(msg)

    perch_int = filters['perchDist']

    PC_titles = {
        'PC01': 'wing lifting',
        'PC02': 'wing spreading',
        'PC03': 'wing sweeping',
        'PC04': 'tail spreading',
        'PC05': 'counter pitching',
        'PC06': 'collective pitching',
        'PC07': 'handwing spreading',
        'PC08': 'M-folding',
        'PC09': 'handwing sweeping',
    }

    PC_names = [f'PC{i:02}' for i in PC_num_list]

    # Re-order PC08 next to PC05 only when both are present
    if 'PC08' in PC_names and 'PC05' in PC_names:
        PC_names.remove('PC08')
        PC_names.insert(PC_names.index('PC05') + 1, 'PC08')

    n = len(PC_names)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig_w = n_cols * 5 / 3
    fig_h = n_rows * 5 / 3

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, squeeze=False
    )
    flat_axes = axes.flatten()

    for ii, pc_name in enumerate(PC_names):
        ax = flat_axes[ii]
        plot_score(scores_df, pc_name, ax, **filters)

        y_min, y_max = ax.get_ylim()
        y_mid = np.round(max(abs(y_min), abs(y_max)) / 2, 2)
        ax.set_yticks([-y_mid, 0, y_mid])

        title = PC_titles.get(pc_name, pc_name)
        ax.set_title(title, fontsize=8, position=(0.5, 0.9))
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(direction='in')
        ax.set_xticks(np.arange(0, -perch_int * 2, -perch_int / 2))
        ax.set_xlim(-perch_int, 0)

    # Hide unused axes
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    fig.text(0.5, -0.02, 'horizontal distance to perch (m)', ha='center')
    fig.tight_layout()

    return fig

def plot_score_multi_distance(scores_df, PC_name, **filters):
    """Plot a single PC score trace at each of four perch distances (5, 7, 9, 12 m).

    Creates a 4x1 stack of subplots with shared axes, one per perch distance.
    Useful for comparing how a particular morphing mode changes with approach
    distance.

    Args:
        scores_df: DataFrame containing PC scores and per-frame metadata.
        PC_name: Name of the PC column to plot (e.g. 'PC01').
        **filters: Keyword arguments forwarded to plot_score(). Must not include
            perchDist, which is set internally for each row.

    Returns:
        Tuple of (fig, axes).

    Raises:
        ValueError: If perchDist is included in filters.
    """
    if 'perchDist' in filters:
        msg = "perchDist should not be in filters"
        raise ValueError(msg)

    perchDist_list = ['5m', '7m', '9m', '12m']

    fig, axes = plt.subplots(4, 1, figsize=(4, 5), sharex=True, sharey=True)

    # Loop through each distance and plot the scores
    for ii, perch in enumerate(perchDist_list):
        ax = axes.flatten()[ii]
        plot_score(scores_df, PC_name, ax, perchDist=perch, **filters)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='y', direction='out')
        ax.tick_params(axis='x', direction='out')

        ax.yaxis.set_label_position("right")
        ax.set_ylabel(
            perchDist_list[ii],
            rotation=0,
            fontsize=8,
            va='center',
            labelpad=10,
        )

        if ii != len(perchDist_list) - 1:
            ax.xaxis.set_ticklabels([])

    # Shared y-axis: use the widest range across all subplots
    global_ymin = min(ax.get_ylim()[0] for ax in axes.flatten())
    global_ymax = max(ax.get_ylim()[1] for ax in axes.flatten())
    axes.flatten()[0].set_ylim(global_ymin, global_ymax)

    # Tight layout with minimal margins; leave room for shared labels on
    # the outer edges.
    fig.subplots_adjust(
        left=0.12, bottom=0.09, right=0.9, top=0.98, hspace=0.1
    )

    # Shared axis labels at outer edges
    fig.text(
        0.03, 0.53, f'{PC_name} score', ha='center', va='center', rotation='vertical'
    )
    fig.text(
        0.5, 0.01, 'horizontal distance to perch (m)', ha='center', va='bottom'
    )

    return fig, axes


def plot_pc_comparison_grid(
    scores_df,
    score_5,
    score_95,
    n_pcs=9,
    alpha=0.1,
    bkgrd_color='white',
    filter_condition=None,
):
    """Create a pairwise scatter-plot grid for exploring PC score correlations.

    Diagonal panels show histograms of each PC score distribution;
    off-diagonal panels show scatter plots of each PC pair. Useful for
    identifying non-independence between morphing modes.

    Args:
        scores_df: DataFrame containing PC scores.
        score_5: 5th-percentile scores per PC, used for axis limits.
        score_95: 95th-percentile scores per PC, used for axis limits.
        n_pcs: Number of PCs to include. Defaults to 9.
        alpha: Scatter-point opacity. Defaults to 0.1.
        bkgrd_color: Background colour for each panel. Defaults to 'white'.
        filter_condition: Optional boolean mask to subset scores_df before
            plotting. If None, all rows are used.

    Returns:
        Figure containing the pairwise comparison grid.
    """
    # Apply filter if provided
    if filter_condition is not None:
        filtered_scores = scores_df[filter_condition].copy()
    else:
        filtered_scores = scores_df.copy()

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(
        n_pcs,
        n_pcs,
        figsize=(20, 20),
        sharex='col',
        sharey='row',
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05},
    )

    # Define colors for each PC
    colour_list = [
        '#B5E675',
        '#6ED8A9',
        '#51B3D4',
        '#4579AA',
        '#F19EBA',
        '#BC96C9',
        '#917AC2',
        '#BE607F',
        '#624E8B',
        '#888888',
        '#888888',
        '#888888',
    ]

    # Get PC column names
    PC_cols = [f'PC{i:02}' for i in range(1, n_pcs + 1)]

    # Loop through each pair of PCs
    for i in range(n_pcs):
        for j in range(n_pcs):
            # Get the PC names
            pc_i = PC_cols[i]
            pc_j = PC_cols[j]

            # Get the axis
            ax = axs[i, j]

            # Set background color
            ax.set_facecolor(bkgrd_color)

            # If on the diagonal, plot histogram
            if i == j:
                ax.hist(filtered_scores[pc_i], bins=30, color=colour_list[i], alpha=0.7)
                ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.8)
                ax.set_title(f'{pc_i}', fontsize=10)
            else:
                # Plot scatter of PC_i vs PC_j
                ax.scatter(
                    filtered_scores[pc_j],
                    filtered_scores[pc_i],
                    marker='o',
                    s=0.2,
                    c=colour_list[i],
                    alpha=alpha,
                    edgecolors='none',
                )

                # Get axis limits
                min_val_i = score_5[pc_i]
                max_val_i = score_95[pc_i]
                min_val_j = score_5[pc_j]
                max_val_j = score_95[pc_j]

                # Set axis limits
                ax.set_xlim(min_val_j, max_val_j)
                ax.set_ylim(min_val_i, max_val_i)

                # Add grid
                ax.grid(True, linestyle=':', alpha=0.3)

            # Only show labels on the edges
            if j == 0:
                ax.set_ylabel(f'{pc_i}', fontsize=10)
            if i == n_pcs - 1:
                ax.set_xlabel(f'{pc_j}', fontsize=10)

            # Remove tick labels for inner plots
            if j != 0 and i != n_pcs - 1:
                ax.set_yticklabels([])
                ax.set_xticklabels([])

    plt.suptitle('Comparison of PC Scores', fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_score_multi_bird(scores_df, PC_name, birdname_list=None, **filters):
    """Plot a single PC score trace separately for each individual hawk.

    Creates a 4x1 stack of subplots with shared axes, one per bird. Useful for
    comparing how individual birds differ in their use of a particular morphing
    mode along the approach.

    Args:
        scores_df: DataFrame containing PC scores and per-frame metadata.
        PC_name: Name of the PC column to plot (e.g. 'PC01').
        birdname_list: Bird names to show, one per row. Defaults to
            ['Drogon', 'Ruby', 'Toothless', 'Charmander'].
        **filters: Keyword arguments forwarded to plot_score().

    Returns:
        Tuple of (fig, axes).
    """
    if birdname_list is None:
        birdname_list = ['Drogon', 'Ruby', 'Toothless', 'Charmander']
    fig, axes = plt.subplots(4, 1, figsize=(5, 4), sharex=True, sharey=True)

    for ii, bird in enumerate(birdname_list):
        ax = axes.flatten()[ii]

        plot_score(scores_df, PC_name, ax, hawkname=bird, **filters)

        # Turn off frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # axes.flatten()[ii].spines['left'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='y', direction='out')
        ax.tick_params(axis='x', direction='in')




        # Y label on the right
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(bird, rotation=0, fontsize=8, va='center', labelpad=10)

        # X axis limits
        # ax.set_xlim(-12,0)

        # if ii == 3:
        #     ax.text(1.3,0.5,"perchDist_list[ii]", ha='center', va='center',
        #             fontsize=9, transform=ax.transAxes)

        if ii != 3:
            ax.xaxis.set_ticklabels([])


    # Set y-axis to span all hawks (sharey means setting one sets all)
    global_ymin = min(ax.get_ylim()[0] for ax in axes.flatten())
    global_ymax = max(ax.get_ylim()[1] for ax in axes.flatten())
    padding = (global_ymax - global_ymin) * 0.05
    axes.flatten()[0].set_ylim(global_ymin - padding, global_ymax + padding)

    fig.text(0.5, 1, 'horizontal distance to perch (m)', ha='center')
    fig.text(0, 0.5, 'PC score', ha='center', rotation='vertical')

    fig.tight_layout()

    return fig, axes
