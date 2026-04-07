"""Condition-comparison score plots (obstacle, weight, naive vs experienced)."""

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from ..data_filtering import filter_by
from ..pca_scores import get_binned_scores
from .scores import plot_score


def plot_score_obstacle_control(
    scores_df,
    PC_name,
    hawkname_list=("Drogon", "Toothless", "Charmander", "Ruby"),
    **filters,
):
    """Plot binned PC score traces comparing obstacle and control flights for each hawk.

    Creates a 1xN row of subplots (one per hawk). Within each subplot the control
    flight trace is shown as a dotted line with lighter shading and the obstacle
    flight as a solid line, so deviations in wing shape during obstacle avoidance
    are immediately visible.

    Args:
        scores_df: DataFrame containing PC scores and per-frame metadata.
        PC_name: Name of the PC column to plot (e.g. 'PC01').
        hawkname_list: Hawks to include, one subplot each. Defaults to all four
            hawks from the 2020 experiment.
        **filters: Additional keyword filters forwarded to plot_score. Must not
            include 'obstacle', which is managed internally.

    Returns:
        Tuple of (fig, axes).

    Raises:
        ValueError: If 'obstacle' is included in filters.
    """
    if 'obstacle' in filters:
        msg = "Obstacle should not be in filters"
        raise ValueError(msg)

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


def plot_score_weight_control(
    scores_df,
    PC_name,
    hawkname_list=("Drogon", "Toothless", "Charmander", "Ruby"),
    **filters,
):
    """Plot binned PC score traces comparing weight-loaded and control flights
    for each hawk.

    Creates a 1xN row of subplots (one per hawk). Control flights are shown as
    dotted lines; weight-loaded flights as solid lines. This reveals how the
    added IMU weight shifts wing-shape strategies during landing.

    Args:
        scores_df: DataFrame containing PC scores and per-frame metadata.
        PC_name: Name of the PC column to plot (e.g. 'PC01').
        hawkname_list: Hawks to include, one subplot each. Defaults to all four
            hawks from the 2020 experiment.
        **filters: Additional keyword filters forwarded to plot_score. Must not
            include 'IMU', which is managed internally.

    Returns:
        Tuple of (fig, axes).

    Raises:
        ValueError: If 'IMU' is included in filters.
    """
    if 'IMU' in filters:
        msg = "IMU/Weight should not be in filters"
        raise ValueError(msg)

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


def plot_score_naive_control(
    scores_df,
    PC_name,
    hawkname_list=("Drogon", "Toothless", "Rhaegal"),
    **filters,
):
    """Plot binned PC score traces comparing naive (juvenile) and experienced flights.

    Creates a 1xN row of subplots (one per hawk). 2017 (naive/juvenile) flights
    are drawn as dotted lines; 2020 (experienced) flights as solid lines. This
    reveals how wing-shape strategies change with experience over the hawks'
    first years of landing practice.

    Args:
        scores_df: DataFrame containing PC scores and per-frame metadata.
        PC_name: Name of the PC column to plot (e.g. 'PC01').
        hawkname_list: Hawks to include; must have data in both years.
            Defaults to the three hawks with 2017 data.
        **filters: Additional keyword filters forwarded to plot_score. Must not
            include 'obstacle', which is managed internally.

    Returns:
        Tuple of (fig, axes).

    Raises:
        ValueError: If 'obstacle' is included in filters.
    """
    if 'obstacle' in filters:
        msg = "Obstacle should not be in filters"
        raise ValueError(msg)

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
