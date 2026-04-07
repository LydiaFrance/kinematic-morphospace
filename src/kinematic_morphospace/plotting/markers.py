"""Raw marker scatter plots and interactive 3-D visualisations.

Functions here display the distribution of wing-marker positions in 2-D
projections and 3-D interactive scatter plots, useful for quality-checking
labelled marker data and inspecting marker cloud geometry.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt


def plot_raw_markers(ax, x, y, filter=None, colour='k', alpha=0.1, grid=False):
    """Scatter plot of raw 2-D marker positions with standardised axis formatting.

    Plots marker positions as a translucent scatter cloud with fixed axis limits
    and equal aspect ratio. Suitable for overlaying multiple marker groups or
    comparing left/right wings.

    Args:
        ax: Matplotlib Axes object to draw on.
        x: X-coordinate array for each marker observation.
        y: Y-coordinate array for each marker observation, aligned with x.
        filter: Optional boolean mask selecting which observations to include.
            If None, all observations are plotted.
        colour: Scatter point colour string. Defaults to 'k' (black).
        alpha: Scatter point opacity. Defaults to 0.1.
        grid: Whether to display grid lines. Defaults to False.

    Returns:
        Axes object containing the scatter plot.
    """
    if filter is not None:
        ax.scatter(
            x[filter], y[filter], marker='o', s=0.1, c=colour, alpha=alpha,
            edgecolors='none'
        )
    else:
        ax.scatter(
            x, y, marker='o', s=0.1, c=colour, alpha=alpha, edgecolors='none'
        )
    # Have both axes ticks as 0, 0.25, 0.5
    ax.set_xticks(np.arange(-0.5, 0.51, 0.25))
    ax.set_yticks(np.arange(-0.5, 0.51, 0.25))
    # Make axis tick labels smaller font
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    # Make sure grid is behind data
    ax.set_axisbelow(True)
    ax.grid(False)
    # Make grid lines very pale
    if grid:
        ax.grid(True)
        ax.grid(color='0.9')
    # Make tick lengths zero
    ax.tick_params(length=0)
    ax.set_aspect('equal')

    # Change background colour
    ax.set_facecolor('white')

    return ax

def plot_uncorrected_markers(df, bird_configs, fig_size=(10, 20),):
    """Plot three-view scatter plots showing raw marker positions for birds.

    Each bird occupies one row with XZ, XY, and YZ projections of the raw
    (uncorrected) labelled marker data. Useful for verifying that marker
    labelling is consistent across birds before applying any rotation
    correction.

    Args:
        df: DataFrame containing the labelled marker data with BirdID, xyz_1,
            xyz_2, and xyz_3 columns.
        bird_configs: List of dicts, one per bird, each with keys 'bird_id' (int),
            'name' (str), 'filters' (dict of additional filter criteria), and
            optionally 'alpha' (float, defaults to 0.1).
        fig_size: Figure size (width, height) in inches. Defaults to (10, 20).

    Returns:
        Tuple of (fig, axs) where axs is a flat array of subplot Axes.
    """
    n_birds = len(bird_configs)
    fig, axs = plt.subplots(n_birds, 3, figsize=fig_size, sharex=True, sharey=True)
    axs = axs.flatten()

    for idx, config in enumerate(bird_configs):
        # Create filter
        filter_conditions = (df['BirdID'] == config['bird_id'])
        for key, value in config.get('filters', {}).items():
            filter_conditions &= (df[key] == value)

        # Plot three views
        base_idx = idx * 3
        alpha = config.get('alpha', 0.1)

        plot_raw_markers(
            axs[base_idx], df['xyz_1'], df['xyz_3'], filter_conditions,
            grid=True, alpha=alpha
        )
        plot_raw_markers(
            axs[base_idx + 1], df['xyz_1'], df['xyz_2'], filter_conditions,
            grid=True, alpha=alpha
        )
        plot_raw_markers(
            axs[base_idx + 2], df['xyz_2'], df['xyz_3'], filter_conditions,
            grid=True, alpha=alpha
        )

        axs[base_idx].set_title(config['name'])
        print(f"{config['name']} Number of points: {len(df[filter_conditions])}")

    return fig, axs

def plot_bird_marker_comparisons(
    frame_info_df, marker_data, birds_config, fig_size=(10, 20), alpha=0.1
):
    """Plot three-view marker scatter plots from array-based marker data.

    Each bird occupies one row with XZ, XY, and YZ projections using the
    first eight markers of the marker_data array (the wing and tail feather
    markers). Used for quality-checking array-format data after
    preprocessing.

    Args:
        frame_info_df: Per-frame metadata DataFrame containing BirdID and any
            additional filter columns specified in birds_config.
        marker_data: Marker position array of shape (n_frames, n_markers, 3).
        birds_config: List of dicts, one per bird, each with keys 'bird_id' (int),
            'name' (str), and 'filters' (dict of extra filter criteria).
        fig_size: Figure size (width, height) in inches. Defaults to (10, 20).
        alpha: Scatter point opacity. Defaults to 0.1.

    Returns:
        Tuple of (fig, axs) where axs is a flat array of subplot Axes.
    """
    n_birds = len(birds_config)
    fig, axs = plt.subplots(n_birds, 3, figsize=fig_size, sharex=True, sharey=True)
    axs = axs.flatten()

    for idx, bird in enumerate(birds_config):
        # Create filter based on BirdID and additional conditions
        filter_conditions = (frame_info_df['BirdID'] == bird['bird_id'])
        for key, value in bird.get('filters', {}).items():
            filter_conditions &= (frame_info_df[key] == value)

        # Plot the three views (XZ, XY, YZ)
        base_idx = idx * 3
        plot_raw_markers(
            axs[base_idx],
            marker_data[filter_conditions, 0:8, 0],
            marker_data[filter_conditions, 0:8, 2],
            grid=True,
            alpha=alpha,
        )
        plot_raw_markers(
            axs[base_idx + 1],
            marker_data[filter_conditions, 0:8, 0],
            marker_data[filter_conditions, 0:8, 1],
            grid=True,
            alpha=alpha,
        )
        plot_raw_markers(
            axs[base_idx + 2],
            marker_data[filter_conditions, 0:8, 1],
            marker_data[filter_conditions, 0:8, 2],
            grid=True,
            alpha=alpha,
        )

        # Set title and print number of points
        axs[base_idx].set_title(bird['name'])
        n_points = len(marker_data[filter_conditions, 0:8, 0].flatten())
        n_seq = len(frame_info_df[filter_conditions]['seqID'].unique())
        print(
            f"{bird['name']} Number of points: {n_points}, "
            f"Number of sequences: {n_seq}"
        )

    return fig, axs


def plot_3d_scatter(x, y, z, time=None):
    """Create an interactive Plotly 3-D scatter plot of wing marker positions.

    Renders marker positions relative to the backpack origin over multiple
    flights. The equal-aspect cube layout preserves the true spatial extent of
    the marker cloud.

    Args:
        x: X-coordinate array, one value per marker observation.
        y: Y-coordinate array, one value per marker observation.
        z: Z-coordinate array, one value per marker observation.
        time: Optional array used to colour markers by time. When None, all
            markers are drawn in black.

    Returns:
        Plotly Figure containing the interactive 3-D scatter plot.
    """
    marker_color = time if time is not None else 'black'
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker={
                'size': 1,
                'color': marker_color,
                'opacity': 0.05,
                'colorscale': 'Viridis' if time is not None else None,
                'showscale': time is not None
            }
        )
    ])

    fig.update_layout(scene={
            'xaxis': {
                'range': [-0.6, 0.6],
                'gridcolor': "rgba(173, 216, 230, 1)",  # Light blue grid lines
                'backgroundcolor': "white",  # White background for the x-axis
                'gridwidth': 1,  # Make grid lines thinner
                'zerolinecolor': "rgba(173, 216, 230, 1)",
                'tickvals': [-0.6, -0.3, 0, 0.3, 0.6],
                'ticktext': ['', '-0.3', '0', '0.3', ''],
                'dtick': 0.1
            },
            'yaxis': {
                'range': [-0.6, 0.6],
                'gridcolor': "rgba(173, 216, 230, 1)",  # Light blue grid lines
                'backgroundcolor': "white",  # White background for the y-axis
                'gridwidth': 1,  # Make grid lines thinner
                'zerolinecolor': "rgba(173, 216, 230, 1)",
                'tickvals': [-0.6, -0.3, 0, 0.3, 0.6],
                'ticktext': ['', '-0.3', '0', '0.3', ''],
                'dtick': 0.1
            },
            'zaxis': {
                'range': [-0.6, 0.6],
                'gridcolor': "rgba(173, 216, 230, 1)",  # Light blue grid lines
                'backgroundcolor': "white",  # White background for the z-axis
                'gridwidth': 1,  # Make grid lines thinner
                'zerolinecolor': "rgba(173, 216, 230, 1)",
                'tickvals': [-0.6, -0.3, 0, 0.3, 0.6],
                'ticktext': ['', '-0.3', '0', '0.3', ''],
                'dtick': 0.1
            },
        'aspectmode': 'cube'
    },
    width=800,
    height=800)

    return fig

def plot_3d_scatter_with_animation(x, y, z, time=None, browser=True):
    """Create an animated Plotly 3-D scatter plot with slow azimuth rotation.

    Builds on plot_3d_scatter() and adds a Plotly animation that rotates the
    camera through a range of azimuth angles, revealing the 3-D structure of the
    marker cloud. A 'Rotate' button triggers the animation.

    Args:
        x: X-coordinate array, one value per marker observation.
        y: Y-coordinate array, one value per marker observation.
        z: Z-coordinate array, one value per marker observation.
        time: Optional array used to colour markers by time. When None, all
            markers are drawn in black.
        browser: When True, the plot opens in the default web browser; when
            False it renders inline in the notebook. Defaults to True.

    Returns:
        Plotly Figure containing the animated 3-D scatter plot.
    """
    # Setup the axes
    fig = plot_3d_scatter(x, y, z, time)

    # Use default elevation (z) for initial elevation
    initial_elevation = 0

    # Define the rotation steps for azimuth angle from 0 to 180 degrees
    # 40 frames for smoother animation
    angles = np.radians(np.linspace(95, 130, 15))

    frames = [
        go.Frame(
            layout={
                'scene_camera': {
                    'eye': {
                        'x': 1.25 * np.cos(angle),
                        'y': 1.25 * np.sin(angle),
                        'z': initial_elevation,
                    }
                }
            }
        )
        for angle in angles
    ]

    # Add frames to the figure
    fig.frames = frames

    # Add animation options
    fig.update_layout(
        updatemenus=[
            {
                'type': "buttons",
                'showactive': False,
                'buttons': [
                    {
                        'label': "Rotate",
                        'method': "animate",
                        'args': [
                            None,
                            {
                                'frame': {'duration': 100, 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate',
                            },
                        ],
                    }
                ],
            }
        ]
    )

    fig.update_layout(
        scene={
            'xaxis': {'tickfont': {'family': 'Andale Mono'}},
            'yaxis': {'tickfont': {'family': 'Andale Mono'}},
            'zaxis': {'tickfont': {'family': 'Andale Mono'}}
        }
    )

    # Set the default renderer to 'browser'
    if browser:
        pio.renderers.default = 'browser'
    else:
        pio.renderers.default = 'notebook'

    pio.show(fig)

    return fig
