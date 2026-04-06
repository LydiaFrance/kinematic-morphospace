"""Cross-species data loading and marker processing for Harvey et al. dataset."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 1. Load data
def load_harvey_data(wing_file, body_file):
    """Load the Harvey et al. wing and body measurement CSV files.

    Args:
        wing_file: Path to the CSV file containing wing marker coordinates.
        body_file: Path to the CSV file containing body morphological measurements.

    Returns:
        Tuple of (wing_df, body_df) as pandas DataFrames.
    """
    wing_df = pd.read_csv(wing_file)
    body_df = pd.read_csv(body_file)
    return wing_df, body_df


# 2. Select the maximum wingspan row for each bird.
def select_max_wingspan_row(df,
                            bird_id_col='BirdID',
                            left_marker='pt8',
                            right_marker='pt12'):
    """Select the row with the maximum wingspan for each bird.

    Wingspan is computed as the Euclidean distance between the left and
    right wingtip markers. This selects the most fully extended posture
    recorded for each individual, which is used as the reference shape.

    Args:
        df: DataFrame containing bird data with 3-D marker coordinates.
        bird_id_col: Column identifying each bird. Defaults to ``'BirdID'``.
        left_marker: Marker name for the left wing tip. Defaults to ``'pt8'``.
        right_marker: Marker name for the right wing tip. Defaults to ``'pt12'``.

    Returns:
        DataFrame with one row per bird, corresponding to the maximum
        wingspan observation.
    """
    df = df.copy()

    # Extract 3D coordinates (X,Y,Z) for left and right wing markers into numpy arrays
    left_coords = df[[f"{left_marker}_X", f"{left_marker}_Y", f"{left_marker}_Z"]].to_numpy()
    right_coords = df[[f"{right_marker}_X", f"{right_marker}_Y", f"{right_marker}_Z"]].to_numpy()

    # Calculate Euclidean distance between left and right markers for each row
    df['wingspan'] = np.linalg.norm(left_coords - right_coords, axis=1)

    max_idx = df.groupby(bird_id_col)['wingspan'].idxmax()
    return df.loc[max_idx].reset_index(drop=True)


# 3. Clean the body data by selecting relevant columns specific to this dataset.
def clean_body_data(body_df):
    """Retain only the morphological columns needed for cross-species analysis.

    Selects a fixed set of columns from the Harvey et al. body dataset,
    including species identity, insertion points, body dimensions, and
    wing/tail measurements.

    Args:
        body_df: Raw body measurement DataFrame from the Harvey et al. dataset.

    Returns:
        DataFrame restricted to the relevant morphological columns.
    """
    columns_to_keep = ['bird_id',
                       'species_common',
                       'x_loc_of_body_max_cm',
                       'x_loc_of_humeral_insert_cm',
                       'y_loc_of_humeral_insert_cm',
                       'z_loc_of_humeral_insert_cm',
                       'body_width_max_cm',
                       'width_at_leg_insert_cm',
                       'head_length_cm',
                       'body_length_cm',
                       'wing_span_cm',
                       'tail_width_cm',
                       'tail_length_cm',
                       'torsotail_length_cm']
    return body_df[columns_to_keep]


# 4. Split bird_id into Species and BirdID.
def split_bird_id(bird_id):
    """Split a compound bird-ID string into species and individual ID.

    The last two underscore-separated parts form the ``BirdID``; all
    preceding parts form the ``Species`` name.

    Args:
        bird_id: Compound bird-ID string (e.g. ``'species_name_bird_01'``).

    Returns:
        Series with keys ``'Species'`` and ``'BirdID'``.
    """
    parts = bird_id.split('_')

    # The last two parts are combined to form the BirdID
    bird_id_new = '_'.join(parts[-2:])

    # The remaining parts are combined to form the Species name
    species = '_'.join(parts[:-2])

    return pd.Series({'Species': species, 'BirdID': bird_id_new})


def process_body_bird_id(body_df, id_col='bird_id'):
    """Extract species and bird ID columns from the body data.

    Filters out rows where the ID column is NaN, splits the compound
    bird-ID string into separate ``Species`` and ``BirdID`` columns, and
    normalises species names to lower case.

    Args:
        body_df: DataFrame containing body measurement data for birds.
        id_col: Name of the column holding the compound bird ID. Defaults
            to ``'bird_id'``.

    Returns:
        Copy of ``body_df`` with added ``Species`` and ``BirdID`` columns.
    """
    body_df = body_df[body_df[id_col].notna()].copy()

    body_df[['Species', 'BirdID']] = body_df[id_col].apply(split_bird_id)

    # Replace specific species names and convert to lowercase so the two
    # DataFrames share consistent species identifiers.
    body_df['Species'] = body_df['Species'].str.replace('COLLI', 'col_liv').str.lower()

    return body_df


# 5. Merge wing and body data on BirdID.
def merge_bird_data(wing_df, body_df, on_col='BirdID'):
    """Merge wing and body measurement DataFrames via a left join.

    Args:
        wing_df: DataFrame containing wing marker data.
        body_df: DataFrame containing body measurement data.
        on_col: Column to join on. Defaults to ``'BirdID'``.

    Returns:
        Merged DataFrame retaining all rows from ``wing_df``.
    """
    return pd.merge(wing_df, body_df, on=on_col, how='left')


# 6. Filter to keep only marker columns (using a base list and a set of marker names)
def filter_marker_columns(df, marker_names, base_columns):
    """Filter a DataFrame to retain only base and marker columns.

    Args:
        df: Input DataFrame.
        marker_names: List of marker name substrings to match against column names.
        base_columns: Columns to always include in the result.

    Returns:
        DataFrame containing ``base_columns`` plus any columns whose names
        contain one of the ``marker_names``.
    """
    marker_cols = [col for col in df.columns if any(marker in col for marker in marker_names)]
    return df[base_columns + marker_cols]


# 7. Set a new origin (e.g., level with the shoulder from a chosen marker).

def set_new_origin_and_axes(df, origin_marker=['pt11', 'pt2'], origin_axes=('x', 'y', 'z'), new_axes=('y', '-x', 'z')):
    """Recentre marker coordinates on a new origin and remap axes.

    Used to align the Harvey et al. coordinate frame with the hawk
    morphospace convention before cross-species comparison.

    Args:
        df: DataFrame containing marker coordinates with ``pt*_X/Y/Z`` columns.
        origin_marker: Marker name (str) or list of two marker names whose
            average position is used as the new origin.
        origin_axes: Tuple of axis names (``'x'``, ``'y'``, ``'z'``) indicating
            which coordinates of the origin marker define the new zero point.
        new_axes: Tuple describing how each output axis maps to the input
            axes. Supports signs, e.g. ``('-x', 'y', 'z')``.

    Returns:
        Copy of ``df`` with all marker coordinates transformed to the new
        origin and axis convention.
    """
    df = df.copy()
    n_rows = len(df)

    # If two markers are provided, use the average of the two as the origin
    if isinstance(origin_marker, list):
        df['avg_origin_X'] = (df[origin_marker[0] + '_X'] + df[origin_marker[1] + '_X']) / 2
        df['avg_origin_Y'] = (df[origin_marker[0] + '_Y'] + df[origin_marker[1] + '_Y']) / 2
        df['avg_origin_Z'] = (df[origin_marker[0] + '_Z'] + df[origin_marker[1] + '_Z']) / 2
        origin_marker = 'avg_origin'

    # Compute origin coordinates from the specified marker and axes
    origin_coords = {
        'x': df[f"{origin_marker}_X"] if 'x' in origin_axes else np.zeros(n_rows),
        'y': df[f"{origin_marker}_Y"] if 'y' in origin_axes else np.zeros(n_rows),
        'z': df[f"{origin_marker}_Z"] if 'z' in origin_axes else np.zeros(n_rows),
    }

    df["origin_x"] = origin_coords['x']
    df["origin_y"] = origin_coords['y']
    df["origin_z"] = origin_coords['z']

    # Identify all marker prefixes
    marker_prefixes = sorted(set(col.split('_')[0] for col in df.columns if col.startswith('pt')))

    # Axis mapping: convert shorthand to operations
    axis_map = {
        'x': lambda row: row['orig_x'],
        'y': lambda row: row['orig_y'],
        'z': lambda row: row['orig_z'],
        '-x': lambda row: -row['orig_x'],
        '-y': lambda row: -row['orig_y'],
        '-z': lambda row: -row['orig_z'],
    }

    # Apply axis transformations
    for marker in marker_prefixes:
        temp_df = pd.DataFrame({
            'orig_x': df[f"{marker}_X"] - df["origin_x"],
            'orig_y': df[f"{marker}_Y"] - df["origin_y"],
            'orig_z': df[f"{marker}_Z"] - df["origin_z"]
        })

        df[f"{marker}_X"] = temp_df.apply(axis_map[new_axes[0]], axis=1)
        df[f"{marker}_Y"] = temp_df.apply(axis_map[new_axes[1]], axis=1)
        df[f"{marker}_Z"] = temp_df.apply(axis_map[new_axes[2]], axis=1)

    return df


# 9. Compute derived markers from existing pt coordinates.

def mirror_marker(df, right_marker, left_marker, x_source, y_source, z_source):
    """Mirror a right-side marker to produce a symmetric left-side marker.

    Creates six new columns in ``df`` by copying the right-side coordinates
    and negating the x-coordinate for the left side, enforcing bilateral
    symmetry around the body midline.

    Args:
        df: DataFrame to modify in place.
        right_marker: Base name for the right-side marker (e.g. ``'right_wingtip'``).
        left_marker: Base name for the left-side marker (e.g. ``'left_wingtip'``).
        x_source: Column name supplying the x coordinate.
        y_source: Column name supplying the y coordinate.
        z_source: Column name supplying the z coordinate.
    """
    df[f"{right_marker}_x"] = df[x_source]
    df[f"{right_marker}_y"] = df[y_source]
    df[f"{right_marker}_z"] = df[z_source]

    df[f"{left_marker}_x"] = -df[x_source]
    df[f"{left_marker}_y"] = df[y_source]
    df[f"{left_marker}_z"] = df[z_source]


def compute_derived_markers(df):
    """Compute derived bilateral markers from existing point coordinates.

    Creates mirrored left/right columns for wingtip, primary, secondary,
    tail-tip, tail-base, and shoulder markers, plus a hood marker.
    Primary markers are averaged from two source points to approximate
    the primary feather insertion region.

    Args:
        df: DataFrame containing the original ``pt*_X/Y/Z`` columns.

    Returns:
        Copy of ``df`` with derived marker columns added.
    """
    df = df.copy()

    # Right wingtip from pt9
    mirror_marker(df, 'right_wingtip', 'left_wingtip', 'pt9_X', 'pt9_Y', 'pt9_Z')

    # Primary markers: average of pt8 and pt4
    df['primary_avg_x'] = (df['pt8_X'] + df['pt4_X']) / 2
    df['primary_avg_y'] = (df['pt8_Y'] + df['pt4_Y']) / 2
    df['primary_avg_z'] = (df['pt8_Z'] + df['pt4_Z']) / 2
    mirror_marker(df, 'right_primary', 'left_primary', 'primary_avg_x', 'primary_avg_y', 'primary_avg_z')

    # Secondary markers from pt10
    mirror_marker(df, 'right_secondary', 'left_secondary', 'pt10_X', 'pt10_Y', 'pt10_Z')

    # Tail markers: tail tip from pt11 with tail length adjustment
    df['tailtip_x'] = df['pt11_X']
    df['tailtip_y'] = df['pt11_Y'] - (df['tail_length_cm'] / 100)
    df['tailtip_z'] = df['pt11_Z']
    mirror_marker(df, 'right_tailtip', 'left_tailtip', 'tailtip_x', 'tailtip_y', 'tailtip_z')

    # Tail base from pt11
    mirror_marker(df, 'right_tailbase', 'left_tailbase', 'pt11_X', 'pt11_Y', 'pt11_Z')

    # Shoulder markers from pt2
    mirror_marker(df, 'right_shoulder', 'left_shoulder', 'pt2_X', 'pt2_Y', 'pt2_Z')

    # Hood marker
    df['hood_x'] = 0
    df['hood_y'] = df['head_length_cm'] / 100
    df['hood_z'] = 0

    # Drop temporary intermediate columns
    df.drop(columns=['primary_avg_x', 'primary_avg_y', 'primary_avg_z',
                     'tailtip_x', 'tailtip_y', 'tailtip_z'], errors='ignore', inplace=True)

    return df


def fix_leftright_sign(df):
    """Ensure left markers have negative x-values and right markers have positive x-values.

    Corrects sign errors that can arise when the Harvey et al. data is
    processed, where some birds may have left/right conventions inverted
    relative to the hawk morphospace coordinate frame.

    Args:
        df: DataFrame containing bilateral marker coordinate columns.

    Returns:
        Copy of ``df`` with corrected marker x-coordinate signs.
    """
    df = df.copy()

    marker_names = ['shoulder', 'wingtip', 'primary', 'secondary', 'tailtip', 'tailbase']

    for marker in marker_names:
        left_col = f'left_{marker}_x'
        right_col = f'right_{marker}_x'

        if left_col not in df.columns or right_col not in df.columns:
            logger.debug("Skipping %s — columns not found", marker)
            continue

        mask = (df[left_col] > 0) & (df[right_col] < 0)

        if mask.any():
            logger.debug("Correcting signs for %d rows of %s", mask.sum(), marker)
            df.loc[mask, left_col] *= -1
            df.loc[mask, right_col] *= -1

    return df


def check_and_fix_shoulder_distance(df, tolerance=0.05):
    """Translate markers so that bilateral distances match morphological measurements.

    Adjusts shoulder, wingtip, tail, and tailbase marker positions by
    translating them symmetrically inward or outward until their pairwise
    distances match the corresponding body measurements (``body_width_max_cm``,
    ``wing_span_cm``, ``tail_width_cm``, ``width_at_leg_insert_cm``). This
    corrects for the cadaver-measurement offsets that would otherwise distort
    the morphospace representation.

    Args:
        df: DataFrame with bilateral marker columns and morphological measurement
            columns (``body_width_max_cm``, ``wing_span_cm``, ``tail_width_cm``,
            ``width_at_leg_insert_cm``).
        tolerance: Allowed relative deviation from the expected distance before
            translation is applied. Defaults to 0.05 (5 %).

    Returns:
        Copy of ``df`` with corrected bilateral marker positions.
    """
    df = df.copy()

    # Calculate shoulder distance
    shoulder_distance = np.linalg.norm(
        df[['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z']].values -
        df[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z']].values, axis=1
    )

    expected_distance = df['body_width_max_cm'] / 100  # Convert cm to metres

    deviation = shoulder_distance - expected_distance

    mask = deviation.abs() > tolerance * expected_distance

    if mask.any():
        logger.debug("Translating markers for %d rows", mask.sum())

        for idx in df.index[mask]:
            offset = deviation[idx] / 2  # Half goes to left side, half to right side

            markers_to_adjust = [
                'shoulder', 'wingtip', 'primary', 'secondary', 'tailtip', 'tailbase'
            ]

            for marker in markers_to_adjust:
                df.at[idx, f'left_{marker}_x'] += offset
                df.at[idx, f'right_{marker}_x'] -= offset

    # Adjust tail width
    tailtip_distance = np.linalg.norm(
        df[['left_tailtip_x', 'left_tailtip_y', 'left_tailtip_z']].values -
        df[['right_tailtip_x', 'right_tailtip_y', 'right_tailtip_z']].values, axis=1
    )
    expected_tail_distance = (df['tail_width_cm'] / 100) * 2  # cm -> m
    tail_deviation = tailtip_distance - expected_tail_distance
    tail_mask = tail_deviation.abs() > tolerance * expected_tail_distance

    if tail_mask.any():
        logger.debug("Translating tailtip markers for %d rows", tail_mask.sum())
        for idx in df.index[tail_mask]:
            offset = tail_deviation[idx]
            df.at[idx, 'left_tailtip_x'] += offset
            df.at[idx, 'right_tailtip_x'] -= offset

    # Adjust tailbase width
    tailbase_distance = np.linalg.norm(
        df[['left_tailbase_x', 'left_tailbase_y', 'left_tailbase_z']].values -
        df[['right_tailbase_x', 'right_tailbase_y', 'right_tailbase_z']].values, axis=1
    )
    expected_tail_distance = df['width_at_leg_insert_cm'] / 100  # cm -> m
    expected_tailbase_distance = expected_tail_distance * 2
    tail_deviation = tailbase_distance - expected_tailbase_distance
    tail_mask = tail_deviation.abs() > tolerance * expected_tailbase_distance

    if tail_mask.any():
        logger.debug("Translating tailbase markers for %d rows", tail_mask.sum())
        for idx in df.index[tail_mask]:
            offset = tail_deviation[idx] / 2
            df.at[idx, 'left_tailbase_x'] += offset
            df.at[idx, 'right_tailbase_x'] -= offset

    # Check the tailbase height is in line with the secondary marker in z
    tailbase_z = df['left_tailbase_z']
    secondary_z = df['left_secondary_z']
    deviation = tailbase_z - secondary_z
    mask = deviation.abs() > tolerance * secondary_z
    tailbase_z = df['right_tailbase_z']
    secondary_z = df['right_secondary_z']
    deviation = tailbase_z - secondary_z
    mask = mask | (deviation.abs() > tolerance * secondary_z)
    if mask.any():
        logger.debug("Translating tailbase markers (z) for %d rows", mask.sum())
        for idx in df.index[mask]:
            offset = deviation[idx]
            df.at[idx, 'left_tailbase_z'] -= offset
            df.at[idx, 'right_tailbase_z'] -= offset
            df.at[idx, 'left_tailtip_z'] -= offset
            df.at[idx, 'right_tailtip_z'] -= offset

    # Adjust wingtip width
    wingtip_distance = np.linalg.norm(
        df[['left_wingtip_x', 'left_wingtip_y', 'left_wingtip_z']].values -
        df[['right_wingtip_x', 'right_wingtip_y', 'right_wingtip_z']].values, axis=1
    )
    expected_wingtip_distance = df['wing_span_cm'] / 100  # cm -> m
    wingtip_deviation = wingtip_distance - expected_wingtip_distance
    wingtip_mask = wingtip_deviation.abs() > tolerance * expected_wingtip_distance

    if wingtip_mask.any():
        logger.debug("Translating wingtip markers for %d rows", wingtip_mask.sum())
        for idx in df.index[wingtip_mask]:
            offset = wingtip_deviation[idx] / 2
            markers_to_adjust = [
                'wingtip', 'primary', 'secondary'
            ]

            for marker in markers_to_adjust:
                df.at[idx, f'left_{marker}_x'] += offset
                df.at[idx, f'right_{marker}_x'] -= offset

    # Adjust shoulder width using the wing root distance (pt1-pt2 separation)
    pt1_x = df['pt1_X']
    pt2_x = df['pt2_X']
    distance_x = (pt2_x - pt1_x).abs()

    wing_root_offset = distance_x * 1.2

    df['left_shoulder_x'] += wing_root_offset
    df['right_shoulder_x'] -= wing_root_offset
    df['left_shoulder_z'] = -df['left_shoulder_z']
    df['right_shoulder_z'] = -df['right_shoulder_z']

    df['hood_z'] = df['right_shoulder_z']

    # Increase tailtip width to a more relaxed estimate (double the distance)
    df['left_tailtip_x'] = df['left_tailtip_x'] * 2
    df['right_tailtip_x'] = df['right_tailtip_x'] * 2

    return df


# 11. Integrate coordinates from a DataFrame into a single marker dictionary for Animal3D.
def integrate_dataframe_to_bird3D(df, row_idx=0):
    """Build a marker dictionary from a DataFrame row for use with ``Animal3D``.

    Extracts moving and fixed bilateral marker positions from a single row
    of ``df`` and assembles them into the ``{marker_name: [x, y, z]}`` format
    accepted by ``Animal3D('hawk', data=markers_dict)``.

    Args:
        df: DataFrame with bilateral marker coordinate columns following the
            ``{side}_{marker}_{axis}`` naming convention.
        row_idx: Index of the row to extract. Defaults to 0.

    Returns:
        Dictionary mapping marker names (e.g. ``'left_wingtip'``) to
        ``[x, y, z]`` coordinate lists.
    """
    markers = {}

    moving_marker_names = ["wingtip", "primary", "secondary", "tailtip"]
    fixed_marker_names = ["shoulder", "tailbase", "hood"]

    # Populate moving markers (e.g., left_wingtip, right_wingtip, etc.)
    for marker in moving_marker_names:
        for side in ['left', 'right']:
            x = df.at[row_idx, f"{side}_{marker}_x"]
            y = df.at[row_idx, f"{side}_{marker}_y"]
            z = df.at[row_idx, f"{side}_{marker}_z"]
            markers[f"{side}_{marker}"] = [x, y, z]

    # Populate fixed markers
    for marker in fixed_marker_names:
        for side in ['left', 'right']:
            if f"{side}_{marker}_x" in df.columns:
                x = df.at[row_idx, f"{side}_{marker}_x"]
                y = df.at[row_idx, f"{side}_{marker}_y"]
                z = df.at[row_idx, f"{side}_{marker}_z"]
                markers[f"{side}_{marker}"] = [x, y, z]

    # Add the hood (which only has one side)
    if "hood_x" in df.columns:
        markers["hood"] = [
            df.at[row_idx, "hood_x"],
            df.at[row_idx, "hood_y"],
            df.at[row_idx, "hood_z"]
        ]

    return markers
