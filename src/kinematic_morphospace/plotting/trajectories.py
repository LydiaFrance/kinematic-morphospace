"""Flight trajectory visualisation against horizontal distance to the perch.

Functions here plot raw scatter clouds and per-hawk binned median lines for
flight trajectories across the full set of experimental conditions (perch
distances, years, obstacles, and weights).
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator

from ..data_filtering import filter_by


def plot_trajectory_data(ax,
                        traj_df,
                        x_col,
                        y_col,
                        filter_params,
                        plot_type='scatter',
                        min_samples=None,
                        **kwargs):
    """Plot trajectory data as a scatter cloud or per-hawk binned median lines.

    Applies a filter and then draws either all individual data points as a
    scatter cloud or per-hawk binned medians as coloured lines. The
    fill_between plot type is used to show individual hawk trajectories
    without obscuring the overall pattern.

    Args:
        ax: Matplotlib Axes object to draw on.
        traj_df: DataFrame containing the trajectory data; must include seqID
            and bins columns for the fill_between type.
        x_col: Column name for the x-axis variable.
        y_col: Column name for the y-axis variable.
        filter_params: Dict of filter kwargs forwarded to filter_by().
        plot_type: Either 'scatter' (all points as a single scatter cloud) or
            'fill_between' (per-hawk binned median lines). Defaults to 'scatter'.
        min_samples: Minimum number of data points required per bin to include a
            median estimate. Bins below this threshold are excluded to avoid
            unreliable lines. Only applies to 'fill_between'. Defaults to None
            (all bins included).
        **kwargs: Additional options. Pass print_n_flights=True to print frame
            and flight counts to stdout.
    """
    # Filter the data based on the filter parameters.
    filter = filter_by(traj_df, **filter_params)

    if plot_type == 'scatter':
        x = traj_df[x_col][filter]
        y = traj_df[y_col][filter]

        # Print the number of flights.
        if kwargs.get('print_n_flights'):
            n_flights = len(np.unique(traj_df['seqID'][filter]))
            print(f"n = {len(x)} points, n_flights = {n_flights}")

        # Scatter plot
        ax.scatter(
            x,
            y,
            marker='o',
            s=0.2,
            c='dodgerblue',
            alpha=0.1,
            edgecolors='none',
        )

    elif plot_type == 'fill_between':
        # This plot type separates the data by hawk.
        # Define the hawk colours.
        hawk_colors = {
            "Drogon": "#FC8D62",
            "Rhaegal": "#8DA0CB",
            "Ruby": "#E78AC3",
            "Toothless": "#66C2A5",
            "Charmander": "#A6D854"
        }

        for hawk, hawk_colour in hawk_colors.items():
            # Filter the data by hawk, finds
            # the mean per binned y-axis.
            hawk_filter = filter_by(traj_df, hawkname=hawk, **filter_params)
            filtered_df = traj_df[hawk_filter]

            if filtered_df.empty:
                continue

            binned_data = filtered_df.groupby('bins').agg(
                {y_col: ['count', 'median', 'std']}
            )

            if min_samples is not None:
                binned_data = binned_data[binned_data[y_col]['count'] >= min_samples]

            x_bins = binned_data.index
            y_median = binned_data[y_col]['median']

            # Not currently used, plots the shading between ±1 standard deviation.
            # ax.fill_between(x_bins, y_mean - y_std, y_mean + y_std,
            #               color=hawk_colors[hawk], alpha=0.1)

            # Plots the mean of each bin per hawk.
            ax.plot(
                x_bins, y_median, color=hawk_colour, label=hawk, linewidth=0.5
            )

def plot_traj(
    traj_df,
    x_axis_column='HorzDistance',
    y_axis_column='XYZ_3',
    equal=True,
    print_n_flights=False,
    min_samples=None,
    save_path=None,
):
    """Create an 8x2 grid of trajectory plots covering all experimental conditions.

    Left column shows raw scatter clouds; right column shows per-hawk binned
    medians. Each row corresponds to a different experimental condition
    (perch distance, year, obstacle, and weight combination). Obstacle flights
    are split by turn direction for the median plot.

    Args:
        traj_df: DataFrame containing the trajectory data.
        x_axis_column: Column name for the x-axis variable. Defaults to
            'HorzDistance'.
        y_axis_column: Column name for the y-axis variable. Defaults to 'XYZ_3'.
        equal: When True, enforces equal aspect ratio with preset limits.
            Defaults to True.
        print_n_flights: When True, prints frame and flight counts per condition.
            Defaults to False.
        min_samples: Minimum number of data points required per bin for the
            per-hawk median lines. Defaults to None (all bins included).
        save_path: Base path for saving separate raster (PNG) and vector (PDF)
            hybrid figures. When None, the figure is not saved. Defaults to None.

    Returns:
        Tuple of (fig, axes) where axes is a flat array of 16 Axes.
    """
    # Create the figure and axes.
    # 8 rows, 2 columns, sharex and sharey.
    # figsize is 8x8 inches.
    fig, axes = plt.subplots(8, 2, sharex=True, sharey=True, figsize=(8, 8))
    axes = axes.flatten()

    # Plot configurations for each subplot.
    # Each subplot is a different experimental condition.
    # - The first number is the horizontal distance to the perch.
    # - The second number is the year of the experiment.
    # - The third number is whether there is an obstacle.
    # - The fourth number is whether there is a weight.
    plot_configs = [
        (5, 2017, 0, 0), (7, 2017, 0, 0), (9, 2017, 0, 0), (12, 2017, 0, 0),
        (9, 2020, 0, 0), (9, 2020, 1, 0), (9, 2020, 0, 1), (9, 2020, 1, 1),
    ]

    # Loop through each subplot and plot the data.
    for ii, (perchDist, year, obstacle, weight) in enumerate(plot_configs):
        print(f"Plotting {perchDist}m, {year}, obstacle={obstacle}, weight={weight}")

        # Setup both axes
        setup_trajectory_axis(axes[ii*2], equal)
        setup_trajectory_axis(axes[ii*2+1], equal)

        # Plot scatter on left axis
        plot_trajectory_data(
            axes[ii * 2],
            traj_df,
            x_axis_column,
            y_axis_column,
            {
                'perchDist': perchDist,
                'year': year,
                'obstacle': obstacle,
                'IMU': weight,
            },
            print_n_flights=print_n_flights,
        )

        # Plot per hawk on right axis.
        #   Left and right turns are separated in the obstacle flights
        #   as the mean between them does not make sense.
        if obstacle == 1:
            for turn in ["left", "right"]:
                plot_trajectory_data(
                    axes[ii * 2 + 1],
                    traj_df,
                    x_axis_column,
                    y_axis_column,
                    {
                        'perchDist': perchDist,
                        'year': year,
                        'obstacle': obstacle,
                        'turn': turn,
                        'IMU': weight,
                    },
                    plot_type='fill_between',
                    min_samples=min_samples,
                )
        else:
            plot_trajectory_data(
                axes[ii * 2 + 1],
                traj_df,
                x_axis_column,
                y_axis_column,
                {
                    'perchDist': perchDist,
                    'year': year,
                    'obstacle': obstacle,
                    'turn': 'straight',
                    'IMU': weight,
                },
                plot_type='fill_between',
                min_samples=min_samples,
            )

    # Add legends
    axes[9].legend(fontsize=4, frameon=False)
    axes[1].legend(fontsize=4, frameon=False)

    # Add x-axis label
    axes[-2].set_xlabel('Horizontal distance to perch (m)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the data as PDF and PNG -- the axes and other elements
    # are saved as vector elements, and the data is saved as raster elements.
    # This is to ensure the axes and other elements are the same size in the
    # vector and raster versions.
    if save_path is not None:
        save_hybrid_figure(fig, axes, save_path)

    return fig, axes

def save_hybrid_figure(fig, axes, base_filename, dpi=600):
    """Save a figure as separate raster (PNG) and vector (PDF) files for compositing.

    The two files are designed to be composited in a layout application:
    the PNG contains only the scatter/line data at high DPI, and the PDF
    contains only the axes, labels, and other non-data elements as vectors.
    This produces publication-quality figures where dense data is rasterised
    and text remains sharp at any size.

    Args:
        fig: The Figure to save.
        axes: All Axes in the figure, used to toggle visibility when switching
            between raster and vector passes.
        base_filename: Output path without extension; '_raster.png' and
            '_vector.pdf' suffixes are appended automatically.
        dpi: Resolution for the raster PNG. Defaults to 600.
    """
    # Store original figure size in inches
    _fig_width, _fig_height = fig.get_size_inches()

    # Hide axes elements for raster version
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Save high DPI PNG for scatter plots
    fig.savefig(f"{base_filename}_raster.png",
                dpi=dpi,
                bbox_inches='tight',
                format='png')


    # Show axes elements for vector version
    for ax in axes:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)

    # Hide scatter plots and dense data
    scatter_artists = []
    for ax in axes:
        for artist in ax.collections + ax.lines:
            if isinstance(
                artist,
                (
                    plt.matplotlib.collections.PathCollection,  # scatter
                    plt.matplotlib.collections.PolyCollection,  # fill_between
                ),
            ):
                scatter_artists.append((artist, artist.get_visible()))
                artist.set_visible(False)

    # Save vector elements with same physical size
    fig.savefig(f"{base_filename}_vector.pdf",
                dpi=dpi,  # Use same DPI to maintain size
                bbox_inches='tight',
                format='pdf')

    # Restore visibility
    for artist, visibility in scatter_artists:
        artist.set_visible(visibility)

def setup_trajectory_axis(ax, equal=True):
    """Configure common axis settings for trajectory plots.

    Sets limits, tick locators, and grid styling appropriate for plotting
    flight trajectories against horizontal distance. The equal-aspect preset
    is sized for the standard 12 m approach corridor.

    Args:
        ax: Matplotlib Axes object to configure.
        equal: When True, enforces equal aspect ratio with x-limits -12.5 to
            0.5 m and y-limits -1.4 to 0.7 m. When False, uses wider y-limits
            (-30 to 120) suitable for angle data. Defaults to True.
    """
    if equal:
        ax.set_aspect('equal')
        ax.set_xlim(-12.5, 0.5)
        ax.set_ylim(-1.4, 0.7)

        # Disable autoscaling
        ax.autoscale(enable=False)

        # Configure axis ticks
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(FixedLocator([-1, 0]))
        ax.yaxis.set_minor_locator(FixedLocator([-0.5, 0.5]))

        # Configure grids
        ax.grid(True, alpha=0.2, which='major')
        ax.grid(True, alpha=0.1, which='minor')
    else:
        ax.set_ylim([-30, 120])
        ax.set_xlim([-12.5, 0.5])
        ax.grid(True, alpha=0.2)

def plot_traj_scatter(
    traj_df,
    x_axis_column='HorzDistance',
    y_axis_column='XYZ_3',
    equal=True,
    print_n_flights=False,
    save_path=None,
):
    """Create 8 scatter plots of trajectories across experimental conditions.

    A single-column layout (8x1) where each row corresponds to a different
    distance/year/obstacle/weight combination. Unlike plot_traj(), no
    per-hawk median column is shown — useful for a compact overview figure.

    Args:
        traj_df: DataFrame containing the trajectory data.
        x_axis_column: Column name for the x-axis variable. Defaults to
            'HorzDistance'.
        y_axis_column: Column name for the y-axis variable. Defaults to 'XYZ_3'.
        equal: When True, enforces equal aspect ratio with preset limits.
            Defaults to True.
        print_n_flights: When True, prints frame and flight counts per condition.
            Defaults to False.
        save_path: Base path for saving hybrid raster/vector figures. When None,
            the figure is not saved. Defaults to None.

    Returns:
        Tuple of (fig, axes) where axes is an array of 8 Axes.
    """
    fig, axes = plt.subplots(8, 1, sharex=True, sharey=True, figsize=(4, 6))

    # Plot configurations for each subplot.
    # Each subplot is a different experimental condition.
    # - The first number is the horizontal distance to the perch.
    # - The second number is the year of the experiment.
    # - The third number is whether there is an obstacle.
    # - The fourth number is whether there is a weight.
    plot_configs = [
        (5,  2017, 0, 0),
        (7,  2017, 0, 0),
        (9,  2017, 0, 0),
        (12, 2017, 0, 0),
        (9, 2020, 0, 0),
        (9, 2020, 1, 0),
        (9, 2020, 0, 1),
        (9, 2020, 1, 1),
    ]
    # Fill each subplot with the data.
    for ax, (perchDist, year, obstacle, weight) in zip(
        axes.flatten(), plot_configs, strict=False
    ):
        print(f"Plotting {perchDist}m, {year}, obstacle={obstacle}, weight={weight}")
        setup_trajectory_axis(ax, equal)
        plot_trajectory_data(
            ax,
            traj_df,
            x_axis_column,
            y_axis_column,
            {
                'perchDist': perchDist,
                'year': year,
                'obstacle': obstacle,
                'IMU': weight,
            },
            print_n_flights=print_n_flights,
        )

    # For the last subplot, add the x-axis label.
    axes[-1].set_xlabel('Horizontal distance to perch (m)')
    plt.tight_layout()

    # Save the figure if a path is provided.
    # This saves the figure as a hybrid PNG and PDF.
    if save_path is not None:
        save_hybrid_figure(fig, axes, save_path)

    return fig, axes
