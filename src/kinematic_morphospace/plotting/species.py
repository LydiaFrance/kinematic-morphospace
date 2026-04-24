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
        line={"color": 'grey', "width": 1},
        showlegend=False,
        hoverinfo='skip',
    ))


def plot_bird_markers(df, row_idx=0, show_derived_lines=True):
    """Interactive Plotly 3-D scatter plot of original and derived bird markers.

    Displays cadaver landmark positions alongside their computationally derived
    counterparts for one species at a time. Blue markers are the original Harvey
    cadaver landmarks (pt* columns); red markers are derived positions computed
    by mirroring or offsetting the originals. Thin grey lines optionally show
    the derivation path from each source point to its derived marker.

    Args:
        df: DataFrame with bird marker coordinates. Must contain pt*_X/Y/Z columns
            for original landmarks and derived keyword columns (wingtip_, primary_,
            secondary_, tailtip_, tailbase_, shoulder_, hood_) for computed markers.
        row_idx: Row index of the species to visualise. Defaults to 0.
        show_derived_lines: When True, draws thin grey lines connecting each
            original landmark to its derived marker. Defaults to True.
    """
    row = df.iloc[row_idx]

    # Gather pt markers
    pt_columns = [col for col in df.columns if col.startswith('pt')]
    pt_markers = sorted({col.split('_')[0] for col in pt_columns})
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
    other_markers = [
        col.rsplit('_', 1)[0] for col in df.columns
        if any(kw in col for kw in derived_keywords) and col.endswith('_x')
    ]
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
            marker={"size": 5, "color": 'blue', "opacity": 0.2},
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
            marker={"size": 5, "color": 'red', "opacity": 0.2},
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
        scene={
            "aspectmode": 'cube',
            "xaxis": {"range": [-min_lim, min_lim], "backgroundcolor": "white",
                       "gridcolor": "grey", "gridwidth": 0.5, "showbackground": True,
                       "zerolinecolor": "grey", "dtick": 0.1,
                       "tickvals": [-0.6, -0.3, 0, 0.3, 0.6],
                       "ticktext": ['', '-0.3', '0', '0.3', '']},
            "yaxis": {"range": [-min_lim, min_lim], "backgroundcolor": "white",
                       "gridcolor": "grey", "gridwidth": 0.5, "showbackground": True,
                       "zerolinecolor": "grey", "dtick": 0.1,
                       "tickvals": [-0.6, -0.3, 0, 0.3, 0.6],
                       "ticktext": ['', '-0.3', '0', '0.3', '']},
            "zaxis": {"range": [-min_lim, min_lim], "backgroundcolor": "white",
                       "gridcolor": "grey", "gridwidth": 0.5, "showbackground": True,
                       "zerolinecolor": "grey", "dtick": 0.1,
                       "tickvals": [-0.6, -0.3, 0, 0.3, 0.6],
                       "ticktext": ['', '-0.3', '0', '0.3', '']},
        },
        width=500,
        height=500,
        margin={"l": 10, "r": 10, "t": 10, "b": 10}
    )
    fig.show()


def prepare_long_neck_bird(
    bird3d,
    neck_length_cm: float,
    head_length_cm: float,
    neck_width_cm: float = None,
    colour: str = 'blue',
    neck_threshold: float = 15.0,
):
    """Prepare a long-necked bird for visualisation.

    For species with neck length exceeding the threshold:

    1. Moves the hood marker to the shoulder position, hiding the built-in
       hood triangle that would otherwise extend too far forward.
    2. Returns a Plotly ``Mesh3d`` trace representing a stylised neck and head
       polygon that can be added to the figure.

    For short-necked species (below threshold), returns ``None`` and leaves
    the bird unchanged.

    The neck length is reduced by 1/3 in the polygon to account for the
    natural S-curve of bird necks that we are not modelling.

    Args:
        bird3d: Animal3D object (modified in place if long-necked).
        neck_length_cm: Neck length in centimetres.
        head_length_cm: Head length in centimetres.
        colour: Polygon fill colour. Defaults to ``'blue'``.
        neck_threshold: Minimum neck length (cm) to trigger long-neck
            handling. Defaults to 15.0.

    Returns:
        A ``plotly.graph_objects.Mesh3d`` trace if long-necked, otherwise
        ``None``.

    Example:
        >>> neck_trace = prepare_long_neck_bird(approx_bird3d, 50.0, 21.0)
        >>> fig = plot_plotly_compare([hawk3d, approx_bird3d], colours=['red', 'blue'])
        >>> if neck_trace is not None:
        ...     fig.add_trace(neck_trace)
    """
    if head_length_cm + neck_length_cm <= neck_threshold:
        return None

    # Get shoulder positions
    shoulder_l = bird3d.fixed_markers[0, 0]
    shoulder_r = bird3d.fixed_markers[0, 1]

    shoulder_width = abs(shoulder_r[0] - shoulder_l[0])
    shoulder_y = (shoulder_l[1] + shoulder_r[1]) / 2
    shoulder_z = (shoulder_l[2] + shoulder_r[2]) / 2

    # Move hood to shoulders (hides built-in triangle)
    bird3d.fixed_markers[0, 4, 0] = 0
    bird3d.fixed_markers[0, 4, 1] = shoulder_y
    bird3d.fixed_markers[0, 4, 2] = shoulder_z

    # Neck dimensions — reduce by 1/3 to account for S-curve
    effective_neck = neck_length_cm * 0.67
    head_width = shoulder_width / 3
    if neck_width_cm is not None:
        neck_width = neck_width_cm
    else:
        neck_width = head_width * 0.6

    neck_base_y = shoulder_y + 0.02
    neck_top_y = neck_base_y + (effective_neck / 100)
    head_tip_y = neck_top_y + (head_length_cm / 100)

    # Vertices: shoulders → neck base → neck top → head tip
    x = [
        shoulder_l[0], shoulder_r[0],
        -neck_width / 2, neck_width / 2,
        -neck_width / 2, neck_width / 2,
        0,
    ]
    y = [
        shoulder_y, shoulder_y,
        neck_base_y, neck_base_y,
        neck_top_y, neck_top_y,
        head_tip_y,
    ]
    z = [shoulder_z] * 7

    # Triangle indices
    i = [0, 0, 2, 2, 4]
    j = [1, 3, 3, 5, 5]
    k = [3, 2, 5, 4, 6]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=colour,
        opacity=0.6,
        flatshading=True,
        showlegend=False,
        hoverinfo='skip',
    )
