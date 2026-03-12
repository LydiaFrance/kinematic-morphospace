import numpy as np
from matplotlib import pyplot as plt

from ..pca_scores import get_binned_scores


def plot_score(scores_df,
                PC_name = 'PC01', ax=None, alpha=1, **filters):
    """Plot a single PC score profile against horizontal distance to the perch.

    Bins the scores by horizontal distance and draws the mean score with
    +/-1 standard deviation shading.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and flight metadata.
    PC_name : str, optional
        Name of the PC column to plot (e.g. ``'PC01'``).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. A new figure is created when *None*.
    alpha : float, optional
        Opacity of the mean line.
    **filters
        Keyword arguments forwarded to :func:`~kinematic_morphospace.pca_scores.get_binned_scores`
        for subsetting (e.g. ``hawkname``, ``obstacle``).

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the score plot.
    """

    if ax is None:
        fig, ax = plt.subplots()

    colour_PC_dict = {'PC01': '#B5E675',    'PC02': '#6ED8A9',   'PC03': '#51B3D4',
                    'PC04': '#4579AA',   'PC05': '#F19EBA',   'PC06': '#BC96C9',
                    'PC07': '#917AC2',   'PC08': '#BE607F',   'PC09': '#624E8B',
                    'PC10': '#888888',  'PC11': '#888888',  'PC12': '#888888'}


    # Bins the scores by horizontal distance to the perch
    # Filters provided by user, e.g. only obstacle flights or by individual bird
    binned_info, mean_scores, stdev_scores, med_scores = get_binned_scores(scores_df, **filters)

    score = mean_scores[PC_name]
    stdev = stdev_scores[PC_name]
    horzDist_bins = binned_info['HorzDistance']

    # If any scores are nan, remove that row and the corresponding row in the horzDist_bins and stdev
    nan_rows = np.isnan(stdev)
    score = score[~nan_rows]
    stdev = stdev[~nan_rows]
    horzDist_bins = horzDist_bins[~nan_rows]

    # Provide a zero score axis
    ax.axhline(y=0, color='#333333', linestyle=':', linewidth=0.5)

    # Shade +/-1 standard deviation
    ax.fill_between(horzDist_bins, score-stdev, score+stdev, color=colour_PC_dict[PC_name], alpha=0.4, edgecolor='none')

    # Plot the mean per bin
    ax.plot(horzDist_bins, score, color=colour_PC_dict[PC_name], linewidth=2, alpha=alpha)

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

def plot_score_multi_PCs(scores_df,PC_num_list=range(1,13), **filters):
    """Plot score profiles for multiple PCs in a 3x3 grid.

    Each panel shows one morphing shape mode, with PC08 re-ordered
    next to PC05 for thematic grouping.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and flight metadata.
    PC_num_list : iterable of int, optional
        PC numbers to include (default 1--12).
    **filters
        Keyword arguments forwarded to :func:`plot_score`. Must include
        ``perchDist``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the 3x3 grid.
    """

    # Warn the user to specify a perch distance
    if 'perchDist' not in filters:
        raise ValueError("perchDist should be in filters")

    perch_int = filters['perchDist']


    # PC_names = ["PC" + str(x) for x in PC_num_list]
    PC_names = [f'PC{i:02}' for i in PC_num_list]
    PC_titles = ["wing lifting",
                "wing spreading",
                "wing sweeping",
                "tail spreading",
                "counter pitching",
                "collective pitching",
                "handwing spreading",
                "M-folding",
                "handwing sweeping"]

    # Move PC08 next to PC05
    # This is for grouping the PCs in similar categories.
    if 'PC08' in PC_names:
        PC_names.remove('PC08')
        PC_names.insert(5, 'PC08')

    fig, axes = plt.subplots(3, 3, figsize=(5, 5), sharex=True)
    flat_axes = axes.flatten()

    # Loop through each subplot and plot the scores.
    for ii, ax in enumerate(flat_axes):
        plot_score(scores_df, PC_names[ii], ax, **filters)

        # plot_score already sets data-driven ylim; read it back
        y_min, y_max = ax.get_ylim()

        # Place one tick at the midpoint of each half
        y_mid = np.round(max(abs(y_min), abs(y_max)) / 2, 2)
        ax.set_yticks([-y_mid, 0, y_mid])


        # Set title for each subplot
        ax.set_title(PC_titles[ii], fontsize = 8,position=(0.5, 0.9))

        # Make x axis labels smaller font
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=6)

        # Turn off frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)


        # Make the ticks point inwards
        ax.tick_params(direction='in')
        ax.set_xticks(np.arange(0,-perch_int*2,-perch_int/2))

        # X axis limits
        ax.set_xlim(-perch_int,0)

    fig.text(0.5, -0.02, 'horizontal distance to perch (m)', ha='center')
    fig.tight_layout()

    return fig

def plot_score_multi_distance(scores_df,PC_name, **filters):
    """Plot a single PC score at four perch distances (5, 7, 9, 12 m).

    Creates a 4x1 stack of subplots with shared axes, one per distance.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and flight metadata.
    PC_name : str
        Name of the PC column to plot (e.g. ``'PC01'``).
    **filters
        Keyword arguments forwarded to :func:`plot_score`. Must *not*
        include ``perchDist``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : numpy.ndarray of matplotlib.axes.Axes
        Array of subplot axes.
    """
    if 'perchDist' in filters:
        raise ValueError("perchDist should not be in filters")

    perchDist_list = [ '5m', '7m','9m','12m']

    fig, axes = plt.subplots(4, 1, figsize=(5, 6), sharex=True, sharey=True)

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
        ax.set_ylabel(perchDist_list[ii], rotation=0, fontsize=8, va='center', labelpad=10)

        if ii != len(perchDist_list) - 1:
            ax.xaxis.set_ticklabels([])

    # Shared y-axis: use the widest range across all subplots
    global_ymin = min(ax.get_ylim()[0] for ax in axes.flatten())
    global_ymax = max(ax.get_ylim()[1] for ax in axes.flatten())
    axes.flatten()[0].set_ylim(global_ymin, global_ymax)

    # First use tight_layout to get good spacing between subplots
    plt.tight_layout()

    # Add extra space for labels
    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95)

    # Add labels with adjusted positions
    fig.text(0.06, 0.5, f'{PC_name} score', ha='center', va='center', rotation='vertical')
    fig.text(0.5, 0.02, 'horizontal distance to perch (m)', ha='center', va='bottom')

    return fig, axes


def plot_pc_comparison_grid(scores_df, score_5, score_95, n_pcs=9, alpha=0.1, bkgrd_color='white', filter_condition=None):
    """Create a pairwise scatter-plot grid comparing PC scores.

    Diagonal panels show histograms; off-diagonal panels show scatter
    plots of each PC pair.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores.
    score_5 : pandas.Series
        5th-percentile scores for axis limits per PC.
    score_95 : pandas.Series
        95th-percentile scores for axis limits per PC.
    n_pcs : int, optional
        Number of PCs to include (default 9).
    alpha : float, optional
        Transparency of scatter points.
    bkgrd_color : str, optional
        Background colour of the panels.
    filter_condition : pandas.Series, optional
        Boolean mask to subset ``scores_df`` before plotting.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the grid.
    """
    # Apply filter if provided
    if filter_condition is not None:
        filtered_scores = scores_df[filter_condition].copy()
    else:
        filtered_scores = scores_df.copy()

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(n_pcs, n_pcs, figsize=(20, 20),
                           sharex='col', sharey='row',
                           gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

    # Define colors for each PC
    colour_list = ['#B5E675', '#6ED8A9', '#51B3D4',
                  '#4579AA', '#F19EBA', '#BC96C9',
                  '#917AC2', '#BE607F', '#624E8B',
                  '#888888', '#888888', '#888888']

    # Get PC column names
    PC_cols = [f'PC{i:02}' for i in range(1, n_pcs+1)]

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
                ax.scatter(filtered_scores[pc_j],
                          filtered_scores[pc_i],
                          marker='o', s=0.2, c=colour_list[i], alpha=alpha, edgecolors='none')

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
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_score_multi_bird(scores_df,PC_name, birdname_list = ['Drogon', 'Ruby','Toothless', 'Charmander'], **filters):
    """Plot a single PC score profile for multiple individual birds.

    Creates a 4x1 stack of subplots, one per bird, with shared axes.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and flight metadata.
    PC_name : str
        Name of the PC column to plot (e.g. ``'PC01'``).
    birdname_list : list of str, optional
        Bird names to plot, one per row.
    **filters
        Keyword arguments forwarded to :func:`plot_score`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : numpy.ndarray of matplotlib.axes.Axes
        Array of subplot axes.
    """

    fig, axes = plt.subplots(4,1,figsize=(5, 4), sharex=True, sharey=True)

    for ii, bird in enumerate(birdname_list):
        ax = axes.flatten()[ii]

        plot_score(scores_df,  PC_name, ax, hawkname=bird, **filters)

     # Turn off frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # axes.flatten()[ii].spines['left'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='y',direction='out')
        ax.tick_params(axis='x',direction='in')




    # Y label on the right
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(bird, rotation=0, fontsize = 8, va = 'center', labelpad = 10)

        # X axis limits
        # ax.set_xlim(-12,0)

        # if ii == 3:
        #     ax.text(1.3,0.5,"perchDist_list[ii]", ha='center', va='center', fontsize=9, transform=ax.transAxes)

        if ii != 3:
            ax.xaxis.set_ticklabels([])


    # Set y-axis to span all hawks (sharey means setting one sets all)
    global_ymin = min(ax.get_ylim()[0] for ax in axes.flatten())
    global_ymax = max(ax.get_ylim()[1] for ax in axes.flatten())
    axes.flatten()[0].set_ylim(global_ymin, global_ymax)

    fig.text(0.5, 1, 'horizontal distance to perch (m)', ha='center')
    fig.text(0,0.5, 'PC score', ha='center', rotation='vertical')

    fig.tight_layout()

    return fig, axes
