"""Cross-species marker visualisation using Plotly."""
import plotly.graph_objects as go


def _get_coords(row, prefix, uppercase=False):
    """Return (x, y, z) for a marker prefix, or None if columns missing."""
    if uppercase:
        cols = (f"{prefix}_X", f"{prefix}_Y", f"{prefix}_Z")
    else:
        cols = (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z")
    if all(c in row.index for c in cols):
        return row[cols[0]], row[cols[1]], row[cols[2]]
    return None


def _add_derivation_line(fig, row, pt_name, derived_name, pt_uppercase=True):
    """Add a thin grey line from an original marker to its derived marker."""
    src = _get_coords(row, pt_name, uppercase=pt_uppercase)
    dst = _get_coords(row, derived_name, uppercase=False)
    if src is None or dst is None:
        return
    fig.add_trace(go.Scatter3d(
        x=[src[0], dst[0]], y=[src[1], dst[1]], z=[src[2], dst[2]],
        mode='lines',
        line=dict(color='grey', width=1),
        showlegend=False,
        hoverinfo='skip',
    ))


def plot_bird_markers(df, row_idx=0, show_derived_lines=True):
    """Interactive 3D scatter plot of original and derived bird markers.

    Blue markers are the original Harvey cadaver landmarks.  Red markers
    are derived (mirrored / offset) positions.  Thin grey lines show the
    derivation path from each source point to its derived marker.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with bird marker coordinates (pt* columns for originals,
        derived keyword columns for computed markers).
    row_idx : int
        Row index to visualise.
    show_derived_lines : bool
        If True (default), draw thin grey lines showing how each derived
        marker was computed from the original landmarks.
    """
    row = df.iloc[row_idx]

    # Gather pt markers
    pt_columns = [col for col in df.columns if col.startswith('pt')]
    pt_markers = sorted(set([col.split('_')[0] for col in pt_columns]))
    x_pt, y_pt, z_pt, labels_pt = [], [], [], []
    for marker in pt_markers:
        coords = _get_coords(row, marker, uppercase=True)
        if coords:
            x_pt.append(coords[0])
            y_pt.append(coords[1])
            z_pt.append(coords[2])
            labels_pt.append(marker)

    # Gather derived markers
    derived_keywords = ['wingtip_', 'primary_', 'secondary_', 'tailtip_',
                        'tailbase_', 'shoulder_', 'hood_']
    other_markers = [col.rsplit('_', 1)[0] for col in df.columns
                     if any(kw in col for kw in derived_keywords) and col.endswith('_x')]
    x_other, y_other, z_other, labels_other = [], [], [], []
    for marker in other_markers:
        coords = _get_coords(row, marker, uppercase=False)
        if coords:
            x_other.append(coords[0])
            y_other.append(coords[1])
            z_other.append(coords[2])
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

    if show_derived_lines:
        # Derivation lines: source pt → right-side derived marker
        # pt9 → wingtip, pt10 → secondary, pt11 → tailbase → tailtip
        # pt2 → shoulder → hood, pt8 → primary, pt4 → primary
        derivations = [
            ('pt9', 'right_wingtip'),
            ('pt8', 'right_primary'),
            ('pt4', 'right_primary'),
            ('pt10', 'right_secondary'),
            ('pt11', 'right_tailbase'),
            ('pt2', 'right_shoulder'),
        ]
        for pt_name, derived_name in derivations:
            _add_derivation_line(fig, row, pt_name, derived_name)

        # tailbase → tailtip (derived → derived, both lowercase)
        _add_derivation_line(fig, row, 'right_tailbase', 'right_tailtip',
                             pt_uppercase=False)
        # shoulder → hood (derived → derived)
        _add_derivation_line(fig, row, 'right_shoulder', 'hood',
                             pt_uppercase=False)

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
