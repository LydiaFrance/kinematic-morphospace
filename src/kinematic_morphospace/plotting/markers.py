import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio


def plot_raw_markers(ax, x, y, filter=None, colour='k', alpha=0.1,  grid=False):
    """Plot marker positions as a 2-D scatter cloud.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    x : numpy.ndarray
        X coordinates of the markers.
    y : numpy.ndarray
        Y coordinates of the markers.
    filter : numpy.ndarray, optional
        Boolean mask selecting which frames to include.
    colour : str, optional
        Colour of the scatter points.
    alpha : float, optional
        Transparency of the scatter points.
    grid : bool, optional
        Whether to display grid lines.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the scatter plot.
    """
    if filter is not None:
        ax.scatter(x[filter], y[filter], marker='o', s=0.1, c=colour, alpha=alpha, edgecolors='none')
    else:
        ax.scatter(x, y, marker='o', s=0.1, c=colour, alpha=alpha, edgecolors='none')
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

    # Check y is not a numpy


    # if y.name.startswith('rot_xyz_1'):
    #     ax.set_ylabel('x (m)')
    # if y.name.startswith('rot_xyz_2'):
    #     ax.set_ylabel('y (m)')
    # if y.name.startswith('rot_xyz_3'):
    #     ax.set_ylabel('z (m)')

    # if x.name.startswith('rot_xyz_1'):
    #     ax.set_xlabel('x (m)')
    # if x.name.startswith('rot_xyz_2'):
    #     ax.set_xlabel('y (m)')
    # if x.name.startswith('rot_xyz_3'):
    #     ax.set_xlabel('z (m)')

    return ax

def plot_uncorrected_markers(df, bird_configs, fig_size=(10, 20)):
    """Plot a grid of three-view marker comparisons for multiple birds.

    Each bird gets one row with XZ, XY, and YZ projections of the
    raw (uncorrected) labelled marker data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the labelled marker data with ``BirdID``,
        ``xyz_1``, ``xyz_2``, and ``xyz_3`` columns.
    bird_configs : list of dict
        One dict per bird with keys ``'bird_id'`` (int), ``'name'``
        (str), ``'filters'`` (dict), and optionally ``'alpha'`` (float).
    fig_size : tuple of float, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Flat array of subplot axes.
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

        plot_raw_markers(axs[base_idx], df['xyz_1'], df['xyz_3'], filter_conditions, grid=True, alpha=alpha)
        plot_raw_markers(axs[base_idx + 1], df['xyz_1'], df['xyz_2'], filter_conditions, grid=True, alpha=alpha)
        plot_raw_markers(axs[base_idx + 2], df['xyz_2'], df['xyz_3'], filter_conditions, grid=True, alpha=alpha)

        axs[base_idx].set_title(config['name'])
        print(f"{config['name']} Number of points: {len(df[filter_conditions])}")

    return fig, axs

def plot_bird_marker_comparisons(frame_info_df, marker_data, birds_config, fig_size=(10, 20), alpha=0.1):
    """Plot three-view marker comparisons from array-based marker data.

    Each bird gets one row with XZ, XY, and YZ projections, using the
    first eight markers of the ``marker_data`` array.

    Parameters
    ----------
    frame_info_df : pandas.DataFrame
        Per-frame metadata (must contain ``BirdID`` and any additional
        filter columns specified in ``birds_config``).
    marker_data : numpy.ndarray
        Marker coordinates, shape ``(n_frames, n_markers, 3)``.
    birds_config : list of dict
        One dict per bird with keys ``'bird_id'`` (int), ``'name'``
        (str), and ``'filters'`` (dict of extra filter criteria).
    fig_size : tuple of float, optional
        Figure size ``(width, height)`` in inches.
    alpha : float, optional
        Transparency of the scatter points.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Flat array of subplot axes.
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
        plot_raw_markers(axs[base_idx],
                                   marker_data[filter_conditions, 0:8, 0],
                                   marker_data[filter_conditions, 0:8, 2],
                                   grid=True, alpha=alpha)
        plot_raw_markers(axs[base_idx + 1],
                                   marker_data[filter_conditions, 0:8, 0],
                                   marker_data[filter_conditions, 0:8, 1],
                                   grid=True, alpha=alpha)
        plot_raw_markers(axs[base_idx + 2],
                                   marker_data[filter_conditions, 0:8, 1],
                                   marker_data[filter_conditions, 0:8, 2],
                                   grid=True, alpha=alpha)

        # Set title and print number of points
        axs[base_idx].set_title(bird['name'])
        n_points = len(marker_data[filter_conditions, 0:8, 0].flatten())
        n_seq = len(frame_info_df[filter_conditions]['seqID'].unique())
        print(f"{bird['name']} Number of points: {n_points}, Number of sequences: {n_seq}")

    return fig, axs


def plot_3d_scatter(x, y, z, time=None):
    """Create an interactive 3-D scatter plot of wing markers.

    Renders marker positions relative to the backpack over multiple
    flights using Plotly.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates (1-D).
    y : numpy.ndarray
        Y coordinates (1-D).
    z : numpy.ndarray
        Z coordinates (1-D).
    time : numpy.ndarray, optional
        Values used to colour markers by time. When *None*, all
        markers are drawn in black.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive 3-D scatter figure.
    """

    marker_color = time if time is not None else 'black'
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=1,
                color=marker_color,
                opacity=0.05,
                colorscale='Viridis' if time is not None else None,
                showscale=True if time is not None else False
            )
        )
    ])

    fig.update_layout(scene=dict(
            xaxis=dict(
                range=[-0.6, 0.6],
                gridcolor="rgba(173, 216, 230, 1)",  # Light blue grid lines
                backgroundcolor="white",  # White background for the x-axis
                gridwidth=1,  # Make grid lines thinner
                zerolinecolor="rgba(173, 216, 230, 1)",
                tickvals=[-0.6, -0.3, 0, 0.3, 0.6],
                ticktext=['', '-0.3', '0', '0.3', ''],
                dtick=0.1
            ),
            yaxis=dict(
                range=[-0.6, 0.6],
                gridcolor="rgba(173, 216, 230, 1)",  # Light blue grid lines
                backgroundcolor="white",  # White background for the y-axis
                gridwidth=1,  # Make grid lines thinner
                zerolinecolor="rgba(173, 216, 230, 1)",
                tickvals=[-0.6, -0.3, 0, 0.3, 0.6],
                ticktext=['', '-0.3', '0', '0.3', ''],
                dtick=0.1
            ),
            zaxis=dict(
                range=[-0.6, 0.6],
                gridcolor="rgba(173, 216, 230, 1)",  # Light blue grid lines
                backgroundcolor="white",  # White background for the z-axis
                gridwidth=1,  # Make grid lines thinner
                zerolinecolor="rgba(173, 216, 230, 1)",
                tickvals=[-0.6, -0.3, 0, 0.3, 0.6],
                ticktext=['', '-0.3', '0', '0.3', ''],
                dtick=0.1
            ),
        aspectmode='cube'
    ),
    width=800,
    height=800)

    return fig

def plot_3d_scatter_with_animation(x, y, z,
                                time=None,
                                browser=True):
    """Create an animated 3-D scatter plot with azimuth rotation.

    Builds on :func:`plot_3d_scatter` and adds a Plotly animation
    that slowly rotates the camera, demonstrating the 3-D structure of
    the marker cloud.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates (1-D).
    y : numpy.ndarray
        Y coordinates (1-D).
    z : numpy.ndarray
        Z coordinates (1-D).
    time : numpy.ndarray, optional
        Values used to colour markers by time.
    browser : bool, optional
        When *True* the plot opens in the default browser; otherwise it
        renders inline in the notebook.

    Returns
    -------
    plotly.graph_objects.Figure
        The animated 3-D scatter figure.
    """
    # Setup the axes
    fig = plot_3d_scatter(x, y, z, time)

    # Use default elevation (z) for initial elevation
    initial_elevation = 0

    # Define the rotation steps for azimuth angle from 0 to 180 degrees
    angles = np.radians(np.linspace(95, 130, 15))  # 40 frames for smoother animation

    frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.25*np.cos(angle), y=1.25*np.sin(angle), z=initial_elevation))))
              for angle in angles]

    # Add frames to the figure
    fig.frames = frames

    # Add animation options
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Rotate",
                          method="animate",
                          args=[None, dict(frame=dict(duration=100, redraw=True),
                                           fromcurrent=True, mode='immediate')])]
        )]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(tickfont=dict(family='Andale Mono')),
            yaxis=dict(tickfont=dict(family='Andale Mono')),
            zaxis=dict(tickfont=dict(family='Andale Mono'))
        )
    )

    # Set the default renderer to 'browser'
    if browser:
        pio.renderers.default = 'browser'
    else:
        pio.renderers.default = 'notebook'

    pio.show(fig)

    return fig
