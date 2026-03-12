"""PCA component loading and principal-cosine visualisation."""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns


def plot_components_grid(principal_components,
                        marker_names,fig=None, ax=None):
    """Plot PCA loadings as a colour-coded heatmap grid.

    Each column represents one principal component and each row a
    marker coordinate. Absolute loadings are displayed with a
    per-component colour map.

    Parameters
    ----------
    principal_components : numpy.ndarray
        Component matrix, shape ``(n_components, n_markers)``.
    marker_names : list of str
        Names labelling each marker coordinate (row labels).
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into. Created if *None*.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. Created if *None*.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes containing the heatmap.
    """

    # Set the number of principal components to plot
    maxPCs = 12

    # Generate the names for the axes labels
    PC_names = [f'PC{i:02}' for i in range(1, maxPCs+1)]

    # Make a dataframe
    components_df = pd.DataFrame.from_dict(dict(zip(PC_names, np.abs(principal_components))))
    components_df["markers"] = marker_names
    components_df = components_df.set_index("markers")

    # We're going to overlay many grids. All but one column and row
    # will be transparent, giving the rainbow effect overall where each PC
    # is a different colour.
    colour_dict = {'PC01': '#B5E675',    'PC02': '#6ED8A9',   'PC03': '#51B3D4',
                    'PC04': '#4579AA',   'PC05': '#F19EBA',   'PC06': '#BC96C9',
                    'PC07': '#917AC2',   'PC08': '#BE607F',   'PC09': '#624E8B',
                    'PC10': '#888888',  'PC11': '#888888',  'PC12': '#888888'}


    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.set_constrained_layout(True)

    # Loop through each PC and plot the grid
    for PC in colour_dict.keys():
        data = components_df.copy()
        # Make every column except the one we're plotting Nan
        data.loc[:, data.columns != PC] = np.nan

        colour_map = mpl.colors.LinearSegmentedColormap.from_list("", ["white",colour_dict[PC]])

        # Add a colour bar for the PC8
        if PC == 'PC8':
                cbar_ax = fig.add_axes([1.05, 0.698, .05, .2], )

                sns.heatmap(data, annot=False, fmt=".2f", linewidth=0.3,
                    cmap = colour_map, vmin = 0, vmax = 1, cbar_ax = cbar_ax, ax = ax, cbar_kws={"label": "absolute loading"})

        else:
            sns.heatmap(data, annot=False, fmt=".2f",
                        cmap = colour_map, vmin = 0, vmax = 1, linewidth=0.3,
                        cbar=False, ax = ax)


    # Add horizontal and vertical lines to the grid
    ax.axhline(y=0, color='#333333',linewidth=1)
    ax.axhline(y=12, color='#333333',linewidth=1)
    ax.axvline(x=0, color='#333333',linewidth=1)
    ax.axvline(x=13, color='#333333',linewidth=1)
    ax.set(ylabel=None)
    ax.set(xlabel="Component Number")
    ax.set_xticklabels(np.arange(1,13), rotation=0)

        # Vertical lines
    # ax.annotate('>95%', xy=(0.575, 0.915), xycoords='figure fraction')
    ax_right = ax.twiny()
    ax_right.spines["bottom"].set_position(("axes", 1.04))
    ax_right.spines["bottom"].set_linewidth(1.5)
    ax_right.xaxis.set_ticks_position("bottom")
    ax_right.xaxis.set_tick_params(width=1.5, length=6)
    ax_right.spines["bottom"].set_visible(True)
    ax_right.set_xticks([-1, -0.32, 0.16, 0.5])
    ax_right.set_xticklabels(['', '', '', ''])


    ax_right.set_xlim(-1, 1)
    ax.annotate('>96%', xy=(0.1, 1.05), xycoords='axes fraction')
    ax.annotate('>98%', xy=(0.4, 1.05), xycoords='axes fraction')
    ax.annotate('>99.9%', xy=(0.6, 1.05), xycoords='axes fraction')


    # Make the ax square
    # ax.set_aspect('equal')

    return fig, ax

def compare_coeffs_hawks(principal_components,
                         principal_components_dict, colour_before =12, y_label='scaled'):
    """Compare principal cosines across individual birds.

    Creates a 1xN row of principal-cosine heatmaps (one per bird),
    showing how closely each bird's individual PCA aligns with the
    pooled component matrix.

    Parameters
    ----------
    principal_components : numpy.ndarray
        Reference (pooled) component matrix.
    principal_components_dict : dict
        Mapping of bird name to that bird's component matrix.
    colour_before : int, optional
        Number of leading components to colour individually.
    y_label : str, optional
        Label for the y-axis of the first panel.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the row of heatmaps.
    """

    fig = plt.figure(figsize=(8, 2))
    gs = gridspec.GridSpec(1, 5, figure=fig, hspace=0, wspace=0.3)  # Adjust these values as needed

    # Get the list of birds
    hawklist = list(principal_components_dict.keys())
    scaled_word_list = [y_label] + ([""] * (len(hawklist) - 1))

    # Loop through each bird PC result and plot the coefficients
    for ii, bird in enumerate(hawklist):
        ax = fig.add_subplot(gs[ii])
        ax = compare_coeffs_grid(principal_components_dict[bird], f"{bird}", principal_components, scaled_word_list[ii], colour_before =colour_before, fig=fig, ax=ax)


        ax.set_yticks(np.arange(0.5,12.5))  # Ensure there are 12 ticks
        ax.set_yticklabels(range(1, 13), rotation=0, fontsize=6)

        ax.set_xticks(np.arange(0.5,12.5))  # Ensure there are 12 ticks
        ax.set_xticklabels(range(1, 13), rotation=0, fontsize=6)

        # Change x axis label fontsize
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

    # ax = fig.add_subplot(gs[5])
    # ax.axis('off')

    return fig

def compare_coeffs_grid(principal_components,
                        name,
                        second_principal_components,
                        second_name,
                        colour_before =12, fig=None,ax=None):
    """Plot a principal-cosine heatmap between two component matrices.

    Computes the absolute dot product of every pair of components and
    displays the result as a colour-coded grid. A strong diagonal
    indicates that the two PCA solutions share the same subspaces.

    Parameters
    ----------
    principal_components : numpy.ndarray
        First component matrix.
    name : str
        Label for the x-axis (identifies the first matrix).
    second_principal_components : numpy.ndarray
        Second component matrix for comparison.
    second_name : str
        Label for the y-axis (identifies the second matrix).
    colour_before : int, optional
        Number of leading components to colour individually.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.
    """


    def check_data_consistency(data1, data2):
        if np.array_equal(data1, data2):
            print(f"{name} data is consistent.")

    check_data_consistency(principal_components, second_principal_components)


    # Print shapes before slicing
    assert principal_components.shape[0] == second_principal_components.shape[1], "Principal components should be square matrix for this test."
    # print(f"{name} principal_components shape (before slicing): {second_principal_components.shape}")


    # Assert orthonormal
    identity_check_1 = np.dot(principal_components, principal_components.T)
    assert np.allclose(np.eye(principal_components.shape[0]), identity_check_1, atol=1e-6), "Principal components are not orthonormal."

    identity_check_2 = np.dot(second_principal_components, second_principal_components.T)
    assert np.allclose(np.eye(second_principal_components.shape[0]), identity_check_2, atol=1e-6), "Comparison Principal components are not orthonormal."


    # Perform dot product calculation
    dot_product_matrix = np.abs(principal_components @ second_principal_components.T)


    # Check if the diagonal values are close to 1
    diagonal_values = np.diag(dot_product_matrix)
    # print(f"{name} Diagonal Values: {diagonal_values}")
    is_diagonal = np.allclose(diagonal_values, np.ones_like(diagonal_values), atol=1e-6)
    if is_diagonal:
        print(f"{name} diagonal values are close to 1, same matrix.")


    # Slicing for plotting
    maxPCs = 12
    if dot_product_matrix.shape[0] > maxPCs:
        dot_product_matrix = dot_product_matrix[:maxPCs, :maxPCs]

    # Print shapes and content after slicing for plotting
    # print(f"dot_product_matrix shape (after slicing): {dot_product_matrix.shape}")


    # marker_names = markers_df.columns.to_list()
    PC_names = [f'PC{i:02}' for i in range(1, maxPCs+1)]
    components_df = pd.DataFrame.from_dict(dict(zip(PC_names, dot_product_matrix)))
    components_df["asym"] = PC_names
    components_df = components_df.set_index("asym")

    base_colour_dict = {'PC01': '#B5E675',    'PC02': '#6ED8A9',   'PC03': '#51B3D4',
                    'PC04': '#4579AA',   'PC05': '#F19EBA',   'PC06': '#BC96C9',
                    'PC07': '#917AC2',   'PC08': '#BE607F',   'PC09': '#624E8B',
                    'PC10': '#888888',  'PC11': '#888888',  'PC12': '#888888'}

    # Create a new colour_dict based on the colour_before value
    colour_dict = {f'PC{i:02}': base_colour_dict.get(f'PC{i:02}', '#888888') if i <= colour_before else '#888888' for i in range(1, maxPCs + 1)}


    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.set_constrained_layout(True)

        returnAx = False
    else:
        returnAx = True

    for PC in colour_dict.keys():
        data = components_df.copy()
        # Make every column except the one we're plotting Nan
        data.loc[:, data.columns != PC] = np.nan

        colour_map = mpl.colors.LinearSegmentedColormap.from_list("", ["white",colour_dict[PC]])

        if PC == 'PC8':
                cbar_ax = fig.add_axes([1.05, 0.698, .05, .2], )

                sns.heatmap(data, annot=False, fmt=".2f", linewidth=0.3,
                    cmap = colour_map, vmin = 0, vmax = 1, cbar_ax = cbar_ax, ax = ax, cbar_kws={"label": "absolute loading"})

        else:
            sns.heatmap(data, annot=False, fmt=".2f",
                        cmap = colour_map, vmin = 0, vmax = 1, linewidth=0.3,
                        cbar=False, ax = ax)


    ax.axhline(y=0, color='#333333',linewidth=1)
    ax.axhline(y=12, color='#333333',linewidth=1)
    ax.axvline(x=0, color='#333333',linewidth=1)
    ax.axvline(x=12, color='#333333',linewidth=1)
    ax.set(xlabel=name)
    ax.set(ylabel=second_name)
    # ax.invert_yaxis()  # Invert y-axis to match matrix indexing


    ax.set_xticks(np.arange(0.5,12.5))  # Ensure there are 12 ticks
    ax.set_xticklabels(range(1, 13), rotation=0, fontsize=10)

    ax.set_yticks(np.arange(0.5,12.5))  # Ensure there are 12 ticks
    ax.set_yticklabels(range(1, 13), rotation=0, fontsize=10)

    # Flip the y axis
    # ax.invert_yaxis()

    # Make the ax square
    ax.set_aspect('equal')

    # fig.tight_layout()
    if returnAx:
        # plt.show()

        return ax



def plot_compare_components_grid(principal_components,
                                 colour_before=2,fig=None, ax=None):
    """Plot a self-comparison principal-cosine grid.

    Displays the absolute dot-product matrix of the component matrix
    with itself. Off-diagonal values near zero confirm orthogonality;
    leading components are colour-coded up to ``colour_before``.

    Parameters
    ----------
    principal_components : numpy.ndarray
        Component matrix, shape ``(n_components, n_markers)``.
    colour_before : int, optional
        Number of leading components to highlight with individual
        colours (default 2).
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into. Created if *None*.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. Created if *None*.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes containing the heatmap.
    """
    maxPCs = 12
    PC_names = [f'PC{i:02}' for i in range(1, maxPCs+1)]


    # make a dataframe
    components_df = pd.DataFrame.from_dict(dict(zip(PC_names, np.abs(principal_components))))
    components_df["names"] = PC_names
    components_df = components_df.set_index("names")

    colour_dict = {'PC01': '#B5E675',    'PC02': '#6ED8A9',   'PC03': '#51B3D4',
                    'PC04': '#4579AA',   'PC05': '#F19EBA',   'PC06': '#BC96C9',
                    'PC07': '#917AC2',   'PC08': '#BE607F',   'PC09': '#624E8B',
                    'PC10': '#888888',  'PC11': '#888888',  'PC12': '#888888'}



    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.set_constrained_layout(True)

    for PC in colour_dict.keys():
        data = components_df.copy()
        # Make every column except the one we're plotting Nan
        data.loc[:, data.columns != PC] = np.nan

        # Find the number for the current PC
        PC_num = int(PC[2:])
        if PC_num <= colour_before:
            colour_map = mpl.colors.LinearSegmentedColormap.from_list("", ["white",colour_dict[PC]])
        else:
            colour_map = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "#888888"])

        if PC == 'PC12':
                cbar_ax = fig.add_axes([1.05, 0.698, .05, .2], )

                sns.heatmap(data, annot=False, fmt=".2f", linewidth=0.3,
                    cmap = colour_map, vmin = 0, vmax = 1, cbar_ax = cbar_ax, ax = ax, cbar_kws={"label": "absolute loading"})

        else:
            sns.heatmap(data, annot=False, fmt=".2f",
                        cmap = colour_map, vmin = 0, vmax = 1, linewidth=0.3,
                        cbar=False, ax = ax)


    ax.axhline(y=0, color='#333333',linewidth=1)
    ax.axhline(y=12, color='#333333',linewidth=1)
    ax.axvline(x=0, color='#333333',linewidth=1)
    ax.axvline(x=13, color='#333333',linewidth=1)
    ax.set(ylabel=None)
    ax.set(xlabel="Component Number")
    ax.set_xticklabels(np.arange(1,13), rotation=0)
    ax.set_yticklabels(np.arange(1,13), rotation=0)

    # Flip the y axis
    ax.invert_yaxis()

    # Make the ax square
    ax.set_aspect('equal')

    fig = fig if fig is not None else ax.get_figure()

    return fig, ax
