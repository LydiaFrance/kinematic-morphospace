"""Whole-body angle visualisation against horizontal distance."""

import numpy as np
from matplotlib import pyplot as plt

from ..data_filtering import filter_by


def bin_and_plot(ax, x, y, color, label):
    """Bin data by horizontal distance and plot mean +/-2 SD.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x : numpy.ndarray
        Horizontal-distance values.
    y : numpy.ndarray
        Angle values corresponding to *x*.
    color : str
        Line and fill colour.
    label : str
        Legend label for the line.

    Returns
    -------
    None
        The axes are modified in place.
    """

    # Define bin edges
    size_bin = 0.05
    bins = np.arange(-12.2, 0.2, size_bin)
    bins = np.around(bins, 3)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Digitize the x values to find out which bin they fall into
    bin_indices = np.digitize(x, bins) - 1

    # Calculate means and standard deviations for each bin
    bin_means = [np.mean(y[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan
                for i in range(len(bins) - 1)]
    bin_stds = [np.std(y[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan
                for i in range(len(bins) - 1)]

    # Convert to numpy arrays
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    # Plotting
    ax.plot(bin_centers, bin_means, color=color, linewidth=2, label=label)
    ax.fill_between(bin_centers, bin_means - (bin_stds*2), bin_means + (bin_stds*2),
                    color=color, alpha=0.4, edgecolor='none')


def plot_whole_body_angles(info_df, euler_angles):
    """Plot whole-body pitch, roll, and yaw for left, right, and straight turns.

    Creates a 3x3 grid: rows are turn directions, columns are angle
    components. Data is binned by horizontal distance.

    Parameters
    ----------
    info_df : pandas.DataFrame
        Per-frame metadata including ``body_pitch``, ``HorzDistance``,
        and turn-direction columns.
    euler_angles : numpy.ndarray
        Euler angles, shape ``(n_frames, 3)`` with columns
        (pitch, yaw, roll).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : numpy.ndarray of matplotlib.axes.Axes
        Flat array of subplot axes.
    """
    pitch_angles = euler_angles[:, 0]
    yaw_angles = euler_angles[:, 1]
    roll_angles = euler_angles[:, 2]

    # Add pitch from dataframe to euler_angles
    pitch_angles = info_df['body_pitch']+pitch_angles


    # Filtering the data
    filter_left = filter_by(info_df, turn='Left')
    filter_right = filter_by(info_df, turn='Right')
    filter_straight = filter_by(info_df, turn='Straight', year=2020)

    fig, ax = plt.subplots(3, 3, figsize=(5, 5), sharex=True, sharey=False)
    ax = ax.flatten()

    # Left Turn
    x_left = info_df['HorzDistance'][filter_left]
    y_left_roll = roll_angles[filter_left]
    y_left_pitch = pitch_angles[filter_left]
    y_left_yaw = yaw_angles[filter_left]

    bin_and_plot(ax[0], x_left, y_left_pitch, '#CC7FDC', 'Pitch')
    bin_and_plot(ax[1], x_left, y_left_roll, '#F2AA79', 'Roll')
    bin_and_plot(ax[2], x_left, y_left_yaw, '#3E92CC', 'Yaw')

    # Right Turn
    x_right = info_df['HorzDistance'][filter_right]
    y_right_roll = roll_angles[filter_right]
    y_right_pitch = pitch_angles[filter_right]
    y_right_yaw = yaw_angles[filter_right]

    bin_and_plot(ax[3], x_right, y_right_pitch, '#CC7FDC', 'Pitch')
    bin_and_plot(ax[4], x_right, y_right_roll, '#F2AA79', 'Roll')
    bin_and_plot(ax[5], x_right, y_right_yaw, '#3E92CC', 'Yaw')

    # Straight
    x_straight = info_df['HorzDistance'][filter_straight]
    y_straight_roll = roll_angles[filter_straight]
    y_straight_pitch = pitch_angles[filter_straight]
    y_straight_yaw = yaw_angles[filter_straight]

    bin_and_plot(ax[6], x_straight, y_straight_pitch, '#CC7FDC', 'Pitch')
    bin_and_plot(ax[7], x_straight, y_straight_roll, '#F2AA79', 'Roll')
    bin_and_plot(ax[8], x_straight, y_straight_yaw, '#3E92CC', 'Yaw')

    # # Titles and labels
    titles = ['Pitch', 'Roll', 'Yaw']

    for i in range(9):
        ax[i].set_ylim(-30, 30)
        ax[i].hlines(0, -9, 0, linestyle='-', color='grey', linewidth=1)
        ax[i].set_xlim(-9, 0)
        ax[i].set_xticks([-9, -4.5, 0])
        ax[i].set_xticklabels(['-9', '-4.5', '0'], fontsize=8)
        ax[i].set_yticks([-20, 0, 20])
        ax[i].set_yticklabels(['-20', '0', '20'], fontsize=8)

    for i in [0,3,6]:
        ax[i].set_ylim(-10, 100)
        ax[i].set_yticks([0, 45, 90])
        ax[i].set_yticklabels(['0', '45', '90'], fontsize=8)



    turn_list = ['Left Turn', 'Right Turn', 'Control']
    counter = 0
    for i in range(2, 9, 3):
        # Put a text label right of the plot at 0
        ax[i].text(0.5, 0.5, turn_list[counter])
        counter += 1



    for i in range(0, 3):
        ax[i].set_title(titles[i])

    # Set common labels
    fig.text(0.5, 0.04, 'Horizontal Distance to perch (m)', ha='center')
    fig.text(0.04, 0.5, 'Angle (degrees)', va='center', rotation='vertical')

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

    return fig, ax


def plot_angles_by_distance(info_df, euler_angles):
    """Plot whole-body angles for each perch distance (5, 7, 9, 12 m).

    Creates a 4x3 grid: rows are perch distances (2017 data only),
    columns are pitch, roll, and yaw.

    Parameters
    ----------
    info_df : pandas.DataFrame
        Per-frame metadata including ``body_pitch``, ``HorzDistance``,
        ``PerchDistance``, and ``Year`` columns.
    euler_angles : numpy.ndarray
        Euler angles, shape ``(n_frames, 3)`` with columns
        (pitch, yaw, roll).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : numpy.ndarray of matplotlib.axes.Axes
        2-D array of subplot axes, shape ``(4, 3)``.
    """

    pitch_angles = euler_angles[:, 0]
    yaw_angles = euler_angles[:, 1]
    roll_angles = euler_angles[:, 2]

    # Add pitch from dataframe to euler_angles
    pitch_angles = info_df['body_pitch'] + pitch_angles

    # Create figure
    fig, ax = plt.subplots(4, 3, figsize=(8, 8), sharex=True)


    # Define distances and colors
    distances = [5, 7, 9, 12]
    colors = {'pitch': '#CC7FDC', 'roll': '#F2AA79', 'yaw': '#3E92CC'}

    # Plot for each distance
    for row, distance in enumerate(distances):
        # Filter data for specific distance and year
        distance_filter = (info_df['PerchDistance'] == distance) & (info_df['Year'] == 2017)

        x_data = info_df['HorzDistance'][distance_filter]
        y_pitch = pitch_angles[distance_filter]
        y_roll = roll_angles[distance_filter]
        y_yaw = yaw_angles[distance_filter]

        # Plot each angle type
        bin_and_plot(ax[row, 0], x_data, y_pitch, colors['pitch'], 'Pitch')
        bin_and_plot(ax[row, 1], x_data, y_roll, colors['roll'], 'Roll')
        bin_and_plot(ax[row, 2], x_data, y_yaw, colors['yaw'], 'Yaw')

        # Add distance label
        ax[row, 0].text(-11.5, 80, f'{distance}m', fontsize=10)

    # Set titles and labels
    titles = ['Pitch', 'Roll', 'Yaw']
    for col, title in enumerate(titles):
        ax[0, col].set_title(title)

    # Format axes
    for i in range(4):
        for j in range(3):
            ax[i, j].set_xlim(-12, 0)
            ax[i, j].set_xticks([-12, -6, 0])
            ax[i, j].grid(True, alpha=0.3)
            ax[i, j].hlines(0, -12, 0, linestyle='-', color='grey', linewidth=1)

            # Set specific y-limits for pitch vs roll/yaw
            if j == 0:  # Pitch
                ax[i, j].set_ylim(-10, 110)
                ax[i, j].set_yticks([0, 45, 90])
            else:  # Roll and Yaw
                ax[i, j].set_ylim(-30, 30)
                ax[i, j].set_yticks([-20, 0, 20])

    # Set common labels
    fig.text(0.5, 0.04, 'Horizontal Distance to perch (m)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Angle (degrees)', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout()

    return fig, ax
