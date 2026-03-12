"""Condition-comparison score plots (obstacle, weight, naive vs experienced)."""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from ..pca_scores import get_binned_scores
from ..data_filtering import filter_by
from .scores import plot_score


def plot_score_obstacle_control(scores_df, PC_name,
                                 hawkname_list=("Drogon", "Toothless", "Charmander", "Ruby"),
                                 **filters):
    """Plot PC scores comparing obstacle vs control flights for each hawk.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and metadata.
    PC_name : str
        Name of the PC column to plot (e.g. 'PC01').
    hawkname_list : sequence of str
        Hawks to plot.
    **filters
        Additional filters passed to plot_score. Must not include 'obstacle'.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    if 'obstacle' in filters:
        raise ValueError("Obstacle should not be in filters")

    condition_labels = ['Control', 'Obstacle']

    fig, axes = plt.subplots(1, len(hawkname_list), figsize=(12, 2.5), sharex=True)

    for ii, hawk in enumerate(hawkname_list):
        for obs in [0, 1]:
            ax = axes.flatten()[ii]
            if obs == 0:
                plot_score(scores_df, PC_name, ax, obstacle=obs, perchDist=9,
                           year=2020, hawkname=hawk, **filters)
            else:
                plot_score(scores_df, PC_name, ax, obstacle=obs, perchDist=9,
                           turn='Right', year=2020, hawkname=hawk, **filters)

            # Make control line dashed
            if obs == 0:
                ax.lines[-1].set_linestyle(':')
                for collection in ax.collections:
                    if isinstance(collection, mpl.collections.PolyCollection):
                        collection.set_alpha(0.2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='y', direction='out')
            ax.tick_params(axis='x', direction='in')

            ax.yaxis.set_label_position("right")

            # Label at last data point
            last_point = ax.lines[-1].get_xydata()[-1]
            ax.text(last_point[0] + 0.1, last_point[1], condition_labels[obs],
                    ha='left', va='center', fontsize=8)

            # Auto-scale y limits
            min_score = np.min([np.min(ax.lines[-1].get_ydata()),
                                np.min(ax.lines[-2].get_ydata())])
            max_score = np.max([np.max(ax.lines[-1].get_ydata()),
                                np.max(ax.lines[-2].get_ydata())])

            max_score = np.ceil(max_score * 100) / 100
            min_score = np.floor(min_score * 100) / 100
            ax.set_ylim(min_score - 0.04, max_score + 0.04)

            if obs != 3:
                ax.xaxis.set_ticklabels([])

        ax.set_title(hawk, fontsize=8)

    fig.text(0.5, -0.02, 'horizontal distance to perch (m)', ha='center')
    fig.text(0, 0.5, 'PC score', ha='center', rotation='vertical')
    fig.tight_layout()

    return fig, axes


def plot_score_weight_control(scores_df, PC_name,
                               hawkname_list=("Drogon", "Toothless", "Charmander", "Ruby"),
                               **filters):
    """Plot PC scores comparing weight vs control flights for each hawk.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and metadata.
    PC_name : str
        Name of the PC column to plot.
    hawkname_list : sequence of str
        Hawks to plot.
    **filters
        Additional filters. Must not include 'IMU'.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    if 'IMU' in filters:
        raise ValueError("IMU/Weight should not be in filters")

    condition_labels = ['Control', 'Weight']

    fig, axes = plt.subplots(1, len(hawkname_list), figsize=(12, 2.5), sharex=True)

    for ii, hawk in enumerate(hawkname_list):
        for exp in [0, 1]:
            ax = axes.flatten()[ii]
            if exp == 0:
                plot_score(scores_df, PC_name, ax, perchDist=9, year=2020,
                           obstacle=0, IMU=1, hawkname=hawk, **filters)
            else:
                plot_score(scores_df, PC_name, ax, perchDist=9, year=2020,
                           obstacle=0, IMU=0, hawkname=hawk, **filters)

            # Make control line dashed
            if exp == 0:
                ax.lines[-1].set_linestyle(':')
                for collection in ax.collections:
                    if isinstance(collection, mpl.collections.PolyCollection):
                        collection.set_alpha(0.2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='y', direction='out')
            ax.tick_params(axis='x', direction='in')

            ax.yaxis.set_label_position("right")

            last_point = ax.lines[-1].get_xydata()[-1]
            ax.text(last_point[0] + 0.1, last_point[1], condition_labels[exp],
                    ha='left', va='center', fontsize=8)

            min_score = np.min([np.min(ax.lines[-1].get_ydata()),
                                np.min(ax.lines[-2].get_ydata())])
            max_score = np.max([np.max(ax.lines[-1].get_ydata()),
                                np.max(ax.lines[-2].get_ydata())])

            max_score = np.ceil(max_score * 100) / 100
            min_score = np.floor(min_score * 100) / 100
            ax.set_ylim(min_score - 0.04, max_score + 0.04)

            if exp != 3:
                ax.xaxis.set_ticklabels([])

        ax.set_title(hawk, fontsize=8)

    fig.text(0.5, -0.02, 'horizontal distance to perch (m)', ha='center')
    fig.text(0, 0.5, 'PC score', ha='center', rotation='vertical')
    fig.tight_layout()

    return fig, axes


def plot_score_naive_control(scores_df, PC_name,
                              hawkname_list=("Drogon", "Toothless", "Rhaegal"),
                              **filters):
    """Plot PC scores comparing naive (juvenile) vs experienced flights for each hawk.

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PC scores and metadata.
    PC_name : str
        Name of the PC column to plot.
    hawkname_list : sequence of str
        Hawks to plot.
    **filters
        Additional filters. Must not include 'obstacle'.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    if 'obstacle' in filters:
        raise ValueError("Obstacle should not be in filters")

    condition_labels = ['Naive', 'Experienced']

    fig, axes = plt.subplots(1, len(hawkname_list), figsize=(8, 4), sharex=True)

    for ii, hawk in enumerate(hawkname_list):
        has_data = True
        for exp in [0, 1]:
            ax = axes.flatten()[ii]
            if exp == 0:
                plot_score(scores_df, PC_name, ax, obstacle=0, perchDist=9,
                           year=2017, hawkname=hawk, **filters)
            else:
                data_filter = filter_by(scores_df, obstacle=0, perchDist=9,
                                        year=2020, hawkname=hawk, **filters)
                if sum(data_filter) > 0:
                    plot_score(scores_df, PC_name, ax, obstacle=0, perchDist=9,
                               year=2020, hawkname=hawk, **filters)
                else:
                    has_data = False

            _, mean_scores, stdev_scores, _ = get_binned_scores(
                scores_df, obstacle=0, perchDist=9, hawkname=hawk, **filters)
            max_score = np.max([
                np.max(mean_scores[PC_name] + stdev_scores[PC_name]),
                np.max(mean_scores[PC_name] - stdev_scores[PC_name])])
            min_score = np.min([
                np.min(mean_scores[PC_name] + stdev_scores[PC_name]),
                np.min(mean_scores[PC_name] - stdev_scores[PC_name])])

            max_score = np.ceil(max_score * 100) / 100
            min_score = np.floor(min_score * 100) / 100
            ax.set_ylim(min_score - 0.04, max_score + 0.04)

            # Make naive line dashed
            if exp == 0:
                ax.lines[-1].set_linestyle(':')
                for collection in ax.collections:
                    if isinstance(collection, mpl.collections.PolyCollection):
                        collection.set_alpha(0.2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='y', direction='out')
            ax.tick_params(axis='x', direction='in')

            ax.yaxis.set_label_position("right")

            if has_data:
                last_point = ax.lines[-1].get_xydata()[-1]
                ax.text(last_point[0] + 0.1, last_point[1], condition_labels[exp],
                        ha='left', va='center', fontsize=8)

            if exp != 3 and has_data:
                ax.xaxis.set_ticklabels([])

        ax.set_title(hawk, fontsize=8)

    fig.text(0.5, -0.02, 'horizontal distance to perch (m)', ha='center')
    fig.text(0, 0.5, 'PC score', ha='center', rotation='vertical')
    fig.tight_layout()

    return fig, axes
