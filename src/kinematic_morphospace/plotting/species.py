"""Cross-species marker visualisation using Plotly."""
import plotly.graph_objects as go


def plot_bird_markers(df, row_idx=0):
    """
    Interactive 3D scatter plot of original and derived bird markers.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with bird marker coordinates (pt* columns for originals,
        derived keyword columns for computed markers).
    row_idx : int
        Row index to visualise.
    """
    # Gather pt markers
    pt_columns = [col for col in df.columns if col.startswith('pt')]
    pt_markers = sorted(set([col.split('_')[0] for col in pt_columns]))
    x_pt, y_pt, z_pt, labels_pt = [], [], [], []
    for marker in pt_markers:
        x_col = f"{marker}_X"
        y_col = f"{marker}_Y"
        z_col = f"{marker}_Z"
        if all(col in df.columns for col in [x_col, y_col, z_col]):
            x_pt.append(df.iloc[row_idx][x_col])
            y_pt.append(df.iloc[row_idx][y_col])
            z_pt.append(df.iloc[row_idx][z_col])
            labels_pt.append(marker)

    # Gather derived markers
    derived_keywords = ['wingtip_', 'primary_', 'secondary_', 'tailtip_',
                        'tailbase_', 'shoulder_', 'hood_']
    other_markers = [col.rsplit('_', 1)[0] for col in df.columns
                     if any(kw in col for kw in derived_keywords) and col.endswith('_x')]
    x_other, y_other, z_other, labels_other = [], [], [], []
    for marker in other_markers:
        x_col = f"{marker}_x"
        y_col = f"{marker}_y"
        z_col = f"{marker}_z"
        if all(col in df.columns for col in [x_col, y_col, z_col]):
            x_other.append(df.iloc[row_idx][x_col])
            y_other.append(df.iloc[row_idx][y_col])
            z_other.append(df.iloc[row_idx][z_col])
            labels_other.append(marker)

    # Build the plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_pt, y=y_pt, z=z_pt,
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.2),
            hovertemplate=('<b>Point</b>: %{text}<br>'
                           'x: %{x:.3f}<br>'
                           'y: %{y:.3f}<br>'
                           'z: %{z:.3f}<br>'
                           '<extra></extra>'),
            text=labels_pt,
            name='Original Points'
        ),
        go.Scatter3d(
            x=x_other, y=y_other, z=z_other,
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.2),
            hovertemplate=('<b>Point</b>: %{text}<br>'
                           'x: %{x:.3f}<br>'
                           'y: %{y:.3f}<br>'
                           'z: %{z:.3f}<br>'
                           '<extra></extra>'),
            text=labels_other,
            name='Derived Points'
        )
    ])

    min_lim = 0.6
    fig.update_layout(
        title=df.iloc[row_idx]['species_common'],
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[-min_lim, min_lim], backgroundcolor="white",
                       gridcolor="grey", gridwidth=0.5, showbackground=True,
                       zerolinecolor="grey", dtick=0.1,
                       tickvals=[-0.6, -0.3, 0, 0.3, 0.6],
                       ticktext=['', '-0.3', '0', '0.3', '']),
            yaxis=dict(range=[-min_lim, min_lim], backgroundcolor="white",
                       gridcolor="grey", gridwidth=0.5, showbackground=True,
                       zerolinecolor="grey", dtick=0.1,
                       tickvals=[-0.6, -0.3, 0, 0.3, 0.6],
                       ticktext=['', '-0.3', '0', '0.3', '']),
            zaxis=dict(range=[-min_lim, min_lim], backgroundcolor="white",
                       gridcolor="grey", gridwidth=0.5, showbackground=True,
                       zerolinecolor="grey", dtick=0.1,
                       tickvals=[-0.6, -0.3, 0, 0.3, 0.6],
                       ticktext=['', '-0.3', '0', '0.3', '']),
        ),
        width=500,
        height=500,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    fig.show()
