import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from ..pca_scores import get_binned_scores
from ..data_filtering import filter_by

PC_NAMES = {
    "PC01": "wing lifting",
    "PC02": "wing spreading",
    "PC03": "wing sweeping",
    "PC04": "tail spreading",
    "PC05": "counter pitching",
    "PC06": "handwing spreading",
    "PC07": "m-folding",
    "PC08": "collective pitching",
    "PC09": "handwing sweeping",
    "PC10": "",
    "PC11": "",
    "PC12": "",
}


def prepare_heatmap_comparison(scores_df, reference_filters, condition1, condition2):
    """Prepare PC scores for comparison between two conditions.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and metadata.
    reference_filters : str or dict
        Either a bird name (str) for score normalisation, or a dict of filters
        to select reference data.
    condition1, condition2 : dict
        Dictionaries containing filters for each condition to compare.

    Returns
    -------
    condition1_df : pandas.DataFrame
        Filtered scores for first condition.
    condition2_df : pandas.DataFrame
        Filtered scores for second condition.
    score_5 : pandas.Series
        1st percentile scores for scaling.
    score_95 : pandas.Series
        99th percentile scores for scaling.
    """
    # Handle reference data selection
    if isinstance(reference_filters, str):
        reference_filter = filter_by(scores_df, hawkname=reference_filters)
    else:
        reference_filter = filter_by(scores_df, **reference_filters)

    reference_scores_df = scores_df[reference_filter].reset_index(drop=True)

    # Create filters for both conditions
    filter1 = filter_by(scores_df, **condition1)
    filter2 = filter_by(scores_df, **condition2)

    # Get filtered scores
    condition1_df = scores_df[filter1].reset_index(drop=True)
    condition2_df = scores_df[filter2].reset_index(drop=True)

    # Calculate score percentiles
    PC_cols = [f'PC{i:02}' for i in np.arange(1, 13)]
    flying_filter = filter_by(reference_scores_df, horzdist=(-0.7, -8))
    score_95 = reference_scores_df.loc[flying_filter, PC_cols].quantile(0.99)
    score_5 = reference_scores_df.loc[flying_filter, PC_cols].quantile(0.01)

    print(f"Condition 1: Number of frames: {len(condition1_df)}; "
          f"Number of flights: {len(np.unique(condition1_df['seqID']))}")
    print(f"Condition 2: Number of frames: {len(condition2_df)}; "
          f"Number of flights: {len(np.unique(condition2_df['seqID']))}")

    return condition1_df, condition2_df, score_5, score_95


def plot_difference_PC_scores_heatmap(df_control,
                                      df_exp,
                                      PC_cols, score_5, score_95):
    """Plot a heatmap comparing PC scores between control and experimental conditions.

    Parameters
    ----------
    df_control : pandas.DataFrame
        DataFrame containing the control scores.
    df_exp : pandas.DataFrame
        DataFrame containing the experimental scores.
    PC_cols : list
        List of PC columns to plot.
    score_5 : dict
        1st percentile scores for colour scaling.
    score_95 : dict
        99th percentile scores for colour scaling.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    # Convert bins to float when finding common bins
    control_bins = df_control['bins'].astype(float).unique()
    exp_bins = df_exp['bins'].astype(float).unique()
    common_bins = sorted(list(set(control_bins) & set(exp_bins)))

    # Calculate means using float bins
    mean_scores_control = df_control[df_control['bins'].astype(float).isin(common_bins)].pivot_table(
        index='bins', values=PC_cols, aggfunc='mean', observed=True).T
    mean_scores_exp = df_exp[df_exp['bins'].astype(float).isin(common_bins)].pivot_table(
        index='bins', values=PC_cols, aggfunc='mean', observed=True).T

    num_pairs = len(PC_cols)
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(num_pairs, 1, height_ratios=[1] * num_pairs, hspace=0.15)

    # Define desired tick positions (-8, -7, ..., -1)
    desired_ticks = np.arange(-8, 0, 1)

    # Get actual x values from the binned data
    actual_positions = mean_scores_control.columns.astype(float)

    # Find the nearest actual bin index for each desired tick position
    tick_indices = [np.abs(actual_positions - tick).argmin() for tick in desired_ticks]

    for ii, PC in enumerate(PC_cols):
        # Get data for this PC
        control_data = mean_scores_control.loc[PC].values
        exp_data = mean_scores_exp.loc[PC].values

        # Combine control and obstacle data for each PC into a single matrix
        combined_data = np.vstack([exp_data, control_data])

        # Remove the last 5 bins (~0.25 m nearest the perch) where the
        # hawk has typically grabbed the perch and marker data is noisy.
        combined_data = combined_data[:, :-5]

        # Plot the combined matrix
        ax = fig.add_subplot(gs[ii])

        # Set vmin and vmax from the 1st and 99th percentile
        vmin = score_5[PC]
        vmax = score_95[PC]

        # Clip values beyond this range
        combined_data = np.clip(combined_data, vmin, vmax)

        im = ax.imshow(combined_data.astype(float), cmap='Spectral_r', aspect='auto',
                        vmin=vmin, vmax=vmax, extent=[0, len(actual_positions), 0, 2])

        # Apply x-tick positions for every subplot
        ax.set_xticks(tick_indices)
        ax.set_xlim(0, len(actual_positions) - 1)

        obstacle_idx = np.abs(actual_positions - (-4.5)).argmin()
        ax.axvline(x=obstacle_idx, color='black', linewidth=1.5, alpha=0.2)

        if ii == 0:
            cbar_ax = fig.add_axes([0.134, 0.91, 0.1, 0.02])
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Score percentile', rotation=0, fontsize=9)
            cbar.ax.xaxis.set_label_position('top')
            cbar.set_ticks([score_5[PC], 0, score_95[PC]])
            cbar.set_ticklabels(["1%", "", "99%"], rotation=0, fontsize=8)

            ax_right = ax.twinx()
            ax_right.set_yticks([0.25, 0.75])
            ax_right.set_yticklabels(['Control', 'Present'], rotation=0, va='center')
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['bottom'].set_visible(False)

            ax_top = ax.twiny()
            ax_top.set_xticks([0.504])
            ax_top.tick_params(axis='x', length=6, width=1.5)
            ax_top.tick_params(axis='x', pad=8)
            ax_top.set_xticklabels(['Obstacle Position'], rotation=0, va='center',
                                    ha='center', style='italic')

            ax.set_xticklabels([])
            ax.set_xticks([])

        elif ii == len(PC_cols) - 1:
            tick_indices = [np.abs(actual_positions - tick).argmin() for tick in desired_ticks]
            ax.set_xticklabels([f"{x:.0f}" for x in desired_ticks], rotation=45)
            ax.set_xlim(0, len(actual_positions) - 1)

            obstacle_idx = np.abs(actual_positions - (-4.5)).argmin()
            ax.axvline(x=obstacle_idx, color='black', linewidth=1.5, alpha=0.2)
            ax.set_xlabel('Binned Horizontal Distance to Perch (m)')

        else:
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_xticks([])

        # PC label on the left
        ax.set_yticks([1])
        ax.set_yticklabels([PC_NAMES[PC]], rotation=0, va='center', fontsize=6)

        # Thin subplot border
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    plt.show()
    return ax


def plot_PC_score_heatmaps(scores_df, PC_cols, score_5, score_95, score_mid, title):
    """Plot a single heatmap of PC scores binned by horizontal distance.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores with a 'bins' column.
    PC_cols : list
        List of PC column names.
    score_5, score_95 : dict
        Percentile scores for colour scaling per PC.
    score_mid : dict
        Median scores for reference.
    title : str
        Plot title.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    pc_scores = scores_df.pivot_table(index='bins', values=PC_cols, aggfunc='mean',
                                       observed=False).T

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    fig.set_constrained_layout(True)

    for ii, PC in enumerate(PC_cols):
        data = pc_scores.copy()
        data.loc[data.index != PC, :] = np.nan
        im = ax.imshow(data, cmap='Spectral', aspect='auto',
                        vmin=score_5[PC], vmax=score_95[PC])

    ax.set_title(title)
    ax.set_ylabel('Principal Components')
    ax.set_xlabel('Binned Horizontal Distance to Perch (m)')
    ax.set_yticks(ticks=np.arange(len(pc_scores.index)))
    ax.set_yticklabels(np.arange(1, len(pc_scores.index) + 1))

    return ax


def plot_difference_exp_scores_heatmap(df_control,
                                        name_control,
                                        df_exp,
                                        name_exp,
                                        PC_cols, score_5, score_95):
    """Plot a heatmap comparing two experimental conditions side by side.

    Parameters
    ----------
    df_control : pandas.DataFrame
        Control condition scores.
    name_control : str
        Label for control condition.
    df_exp : pandas.DataFrame
        Experimental condition scores.
    name_exp : str
        Label for experimental condition.
    PC_cols : list
        List of PC column names.
    score_5, score_95 : dict
        Percentile scores for colour scaling.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    mean_scores_control = df_control.pivot_table(index='bins', values=PC_cols,
                                                  aggfunc='mean', observed=False).T
    mean_scores_exp = df_exp.pivot_table(index='bins', values=PC_cols,
                                          aggfunc='mean', observed=False).T

    num_pairs = len(PC_cols)
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(num_pairs, 1, height_ratios=[1] * num_pairs, hspace=0.15)

    for ii, PC in enumerate(PC_cols):
        min_len = min(mean_scores_control.shape[1], mean_scores_exp.shape[1])
        control_data = mean_scores_control.loc[PC].values[:min_len]
        exp_data = mean_scores_exp.loc[PC].values[:min_len]

        combined_data = np.vstack([exp_data, control_data])
        # Remove the last 5 bins (~0.25 m nearest the perch) where the
        # hawk has typically grabbed the perch and marker data is noisy.
        combined_data = combined_data[:, :-5]

        ax = fig.add_subplot(gs[ii])
        im = ax.imshow(combined_data, cmap='Spectral_r', aspect='auto',
                        vmin=score_5[PC], vmax=score_95[PC])
        im.set_clim(score_5[PC], score_95[PC])

        ax.set_yticks([0, 1])

        if ii == 0:
            cbar_ax = fig.add_axes([0.134, 0.91, 0.1, 0.02])
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Score percentile', rotation=0, fontsize=9)
            cbar.ax.xaxis.set_label_position('top')
            # This heatmap uses 5th/95th percentile scaling (vs 1st/99th
            # in the obstacle heatmap) because it compares conditions
            # where extreme outliers would compress the colour range.
            cbar.set_ticks([score_5[PC], 0, score_95[PC]])
            cbar.set_ticklabels(["5%", "50%", "95%"], rotation=0, fontsize=8)

            ax_right = ax.twinx()
            ax_right.set_yticks([0.25, 0.75])
            ax_right.set_yticklabels([name_control, name_exp], rotation=0, va='center')
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['bottom'].set_visible(False)

            ax.set_xticklabels([])
            ax.set_xticks([])

        elif ii == len(PC_cols) - 1:
            ax.set_xticks(ticks=np.arange(5, len(mean_scores_exp.columns), 20),
                          labels=mean_scores_exp.columns[5::20], rotation=45)
            ax.set_xlabel('Binned Horizontal Distance to Perch (m)')

        else:
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_xticks([])

        # PC label on the left
        ax.set_yticks([0.5])
        ax.set_yticklabels([PC_NAMES[PC]], rotation=0, va='center', fontsize=6)

        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    plt.show()
    return ax
