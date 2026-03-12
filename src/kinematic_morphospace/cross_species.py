"""Cross-species data loading and marker processing for Harvey et al. dataset."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 1. Load data
def load_harvey_data(wing_file, body_file):
    wing_df = pd.read_csv(wing_file)
    body_df = pd.read_csv(body_file)
    return wing_df, body_df

# 2. Select the maximum wingspan row for each bird.
def select_max_wingspan_row(df,
                            bird_id_col='BirdID',
                            left_marker='pt8',
                            right_marker='pt12'):
    """Select the row with the maximum wingspan for each bird.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing bird data with 3-D marker coordinates.
    bird_id_col : str, optional
        Column identifying each bird (default ``'BirdID'``).
    left_marker : str, optional
        Marker name for the left wing tip (default ``'pt8'``).
    right_marker : str, optional
        Marker name for the right wing tip (default ``'pt12'``).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per bird, corresponding to the maximum
        wingspan observation.
    """
    
    df = df.copy()

    # Extract 3D coordinates (X,Y,Z) for left and right wing markers into numpy arrays
    left_coords = df[[f"{left_marker}_X", f"{left_marker}_Y", f"{left_marker}_Z"]].to_numpy()
    right_coords = df[[f"{right_marker}_X", f"{right_marker}_Y", f"{right_marker}_Z"]].to_numpy()

    # Calculate Euclidean distance between left and right markers for each row
    # This gives us the wingspan measurement for each observation
    df['wingspan'] = np.linalg.norm(left_coords - right_coords, axis=1)
    
    # For each unique bird (grouped by bird_id_col):
    # 1. Find the row with maximum wingspan
    # 2. Keep only that row for each bird
    # 3. Reset the index to clean up after groupby operation
    max_idx = df.groupby(bird_id_col)['wingspan'].idxmax()
    return df.loc[max_idx].reset_index(drop=True)


# 3. Clean the body data by selecting relevant columns specific to this dataset.
def clean_body_data(body_df):
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

    Parameters
    ----------
    bird_id : str
        Compound bird-ID string (e.g. ``'species_name_bird_01'``).

    Returns
    -------
    pd.Series
        Series with keys ``'Species'`` and ``'BirdID'``.
    """
    # Split the bird_id string into parts based on the underscore delimiter
    parts = bird_id.split('_')
    
    # The last two parts are combined to form the BirdID
    bird_id_new = '_'.join(parts[-2:])
    
    # The remaining parts are combined to form the Species name
    species = '_'.join(parts[:-2])
    
    # Return the species and bird ID as a pandas Series
    return pd.Series({'Species': species, 'BirdID': bird_id_new})

def process_body_bird_id(body_df, id_col='bird_id'):
    """Extract species and bird ID columns from the body data.

    Filters out rows where the ID column is NaN, splits the compound
    bird-ID string into separate ``Species`` and ``BirdID`` columns, and
    normalises species names to lower case.

    Parameters
    ----------
    body_df : pd.DataFrame
        DataFrame containing body measurement data for birds.
    id_col : str, optional
        Name of the column holding the compound bird ID (default
        ``'bird_id'``).

    Returns
    -------
    pd.DataFrame
        Copy of *body_df* with added ``Species`` and ``BirdID`` columns.
    """
    # Filter out rows with NaN values in the specified ID column
    body_df = body_df[body_df[id_col].notna()].copy()
    
    # Apply the split_bird_id function to extract species and bird ID
    body_df[['Species', 'BirdID']] = body_df[id_col].apply(split_bird_id)
    
    # Replace specific species names and convert to lowercase
    # In this case, making sure the two dataframes have the same species names
    body_df['Species'] = body_df['Species'].str.replace('COLLI', 'col_liv').str.lower()
    
    return body_df

# 5. Merge wing and body data on BirdID.
def merge_bird_data(wing_df, body_df, on_col='BirdID'):
    """Merge wing and body measurement DataFrames via a left join.

    Parameters
    ----------
    wing_df : pd.DataFrame
        DataFrame containing wing marker data.
    body_df : pd.DataFrame
        DataFrame containing body measurement data.
    on_col : str, optional
        Column to join on (default ``'BirdID'``).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame retaining all rows from *wing_df*.
    """

    return pd.merge(wing_df, body_df, on=on_col, how='left')

# 6. Filter to keep only marker columns (using a base list and a set of marker names)
def filter_marker_columns(df, marker_names, base_columns):
    """Filter a DataFrame to retain only base and marker columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    marker_names : list of str
        Marker name substrings to match against column names.
    base_columns : list of str
        Columns to always include in the result.

    Returns
    -------
    pd.DataFrame
        DataFrame containing *base_columns* plus any columns whose names
        contain one of the *marker_names*.
    """
    marker_cols = [col for col in df.columns if any(marker in col for marker in marker_names)]
    return df[base_columns + marker_cols]

# 7. Set a new origin (e.g., level with the shoulder from a chosen marker).

def set_new_origin_and_axes(df, origin_marker=['pt11','pt2'], origin_axes=('x', 'y', 'z'), new_axes=('y', '-x', 'z')):
    """
    Sets a new origin and remaps axes based on specified parameters.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the marker coordinates.
    - origin_marker (str): The marker used as the new origin (e.g., 'pt2').
    - origin_axes (tuple): Which coordinates of the origin marker to use for (x, y, z).
    - new_axes (tuple): How the new axes should be defined relative to the original ('x', 'y', 'z', '-x', '-y', etc.).

    Returns:
    pd.DataFrame: DataFrame with adjusted coordinates.
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
    
    # Assign the new origin coordinates
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
        # Calculate the original coordinates relative to the chosen origin
        temp_df = pd.DataFrame({
            'orig_x': df[f"{marker}_X"] - df["origin_x"],
            'orig_y': df[f"{marker}_Y"] - df["origin_y"],
            'orig_z': df[f"{marker}_Z"] - df["origin_z"]
        })
        
        # Apply the axis transformations for each axis
        df[f"{marker}_X"] = temp_df.apply(axis_map[new_axes[0]], axis=1)
        df[f"{marker}_Y"] = temp_df.apply(axis_map[new_axes[1]], axis=1)
        df[f"{marker}_Z"] = temp_df.apply(axis_map[new_axes[2]], axis=1)

    return df


# 9. Compute derived markers from existing pt coordinates.

def mirror_marker(df, right_marker, left_marker, x_source, y_source, z_source):
    """
    Helper function to mirror markers across the y-axis.
    
    Parameters:
    - df: The DataFrame containing marker coordinates.
    - right_marker: The base name of the right-side marker.
    - left_marker: The base name of the left-side marker.
    - x_source, y_source, z_source: Column names for the right-side coordinates.
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

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the original ``pt*_X/Y/Z`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with derived marker columns added.
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

    # Drop temporary columns (like primary_avg_x) if not needed
    df.drop(columns=['primary_avg_x', 'primary_avg_y', 'primary_avg_z', 
                     'tailtip_x', 'tailtip_y', 'tailtip_z'], errors='ignore', inplace=True)

    return df

def fix_leftright_sign(df):
    """
    Fixes the sign of the left and right markers in a DataFrame.
    Ensures that left markers have negative x-values and right markers have positive x-values.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing marker coordinates
    
    Returns:
    pd.DataFrame: DataFrame with corrected marker signs
    """
    df = df.copy()  # Create a copy to avoid modifying the original
    
    marker_names = ['shoulder', 'wingtip', 'primary', 'secondary', 'tailtip', 'tailbase']
    
    for marker in marker_names:
        left_col = f'left_{marker}_x'
        right_col = f'right_{marker}_x'

        # Check if columns exist
        if left_col not in df.columns or right_col not in df.columns:
            logger.debug("Skipping %s — columns not found", marker)
            continue

        # Find rows where signs need to be corrected
        # (where left is positive and right is negative)
        mask = (df[left_col] > 0) & (df[right_col] < 0)

        if mask.any():
            logger.debug("Correcting signs for %d rows of %s", mask.sum(), marker)
            # Multiply both columns by -1 where correction is needed
            df.loc[mask, left_col] *= -1
            df.loc[mask, right_col] *= -1
    
    return df

def check_and_fix_shoulder_distance(df, tolerance=0.05):
    """
    Adjusts marker positions so the shoulder distance matches body_width_max_cm by translation.

    Parameters:
    - df: DataFrame with markers and body_width_max_cm.
    - tolerance: Allowed relative deviation before translation is applied.

    For each row, this will:
    - Calculate the current shoulder distance.
    - Calculate the expected distance (body_width_max_cm / 100).
    - Translate markers if deviation exceeds the tolerance.
    """
    df = df.copy()

    # Calculate shoulder distance
    shoulder_distance = np.linalg.norm(
        df[['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z']].values -
        df[['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z']].values, axis=1
    )

    # Calculate the expected distance from the body width
    expected_distance = df['body_width_max_cm'] / 100  # Convert cm to meters

    # Calculate deviation
    deviation = shoulder_distance - expected_distance

    # Identify rows needing translation
    mask = deviation.abs() > tolerance * expected_distance

    if mask.any():
        logger.debug("Translating markers for %d rows", mask.sum())

        # Apply translation to each affected row
        for idx in df.index[mask]:
            offset = deviation[idx] / 2  # Half goes to left side, half to right side

            # Markers to adjust
            markers_to_adjust = [
                'shoulder', 'wingtip', 'primary', 'secondary', 'tailtip', 'tailbase'
            ]

            for marker in markers_to_adjust:
                # Translate left markers to the right
                df.at[idx, f'left_{marker}_x'] += offset

                # Translate right markers to the left
                df.at[idx, f'right_{marker}_x'] -= offset

    
    # Adjust tail width
    # Calculate tailtip distance
    tailtip_distance = np.linalg.norm(
        df[['left_tailtip_x', 'left_tailtip_y', 'left_tailtip_z']].values -
        df[['right_tailtip_x', 'right_tailtip_y', 'right_tailtip_z']].values, axis=1
    )
    expected_tail_distance = (df['tail_width_cm'] / 100)*2  # cm -> m
    tail_deviation = tailtip_distance - expected_tail_distance
    tail_mask = tail_deviation.abs() > tolerance * expected_tail_distance

    if tail_mask.any():
        logger.debug("Translating tailtip markers for %d rows", tail_mask.sum())
        for idx in df.index[tail_mask]:
            offset = tail_deviation[idx]
            df.at[idx, 'left_tailtip_x'] += offset
            df.at[idx, 'right_tailtip_x'] -= offset

    # Adjust tailbase width
    # Calculate tailbase distance
    tailbase_distance = np.linalg.norm(
        df[['left_tailbase_x', 'left_tailbase_y', 'left_tailbase_z']].values -
        df[['right_tailbase_x', 'right_tailbase_y', 'right_tailbase_z']].values, axis=1
    )
    expected_tail_distance = df['width_at_leg_insert_cm'] / 100  # cm -> m
    expected_tailbase_distance = expected_tail_distance*2
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
    # Calculate wingtip distance
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
            # Markers to adjust
            markers_to_adjust = [
                'wingtip', 'primary', 'secondary'
            ]

            for marker in markers_to_adjust:
                # Translate left markers to the right
                df.at[idx, f'left_{marker}_x'] += offset

                # Translate right markers to the left
                df.at[idx, f'right_{marker}_x'] -= offset
    

    # Adjust shoulder width
    # calculate the distance between pt1 and pt2 in x, which tells us the wing root. 
    pt1_x = df['pt1_X']
    pt2_x = df['pt2_X']
    distance_x = (pt2_x - pt1_x).abs()

    # Calculate the offset to apply
    wing_root_offset = distance_x*1.2

    # Apply the offset to the left and right shoulder markers
    df['left_shoulder_x'] += wing_root_offset
    df['right_shoulder_x'] -= wing_root_offset
    df['left_shoulder_z'] = -df['left_shoulder_z']
    df['right_shoulder_z'] = -df['right_shoulder_z']

    df['hood_z'] = df['right_shoulder_z']

    # Increase tailtip width to a more relaxed estimate (double the distance)
    df['left_tailtip_x'] = df['left_tailtip_x']*2
    df['right_tailtip_x'] = df['right_tailtip_x']*2

    return df

# 11. Integrate coordinates from a DataFrame into a single marker dictionary for Animal3D.
def integrate_dataframe_to_bird3D(df, row_idx=0):
    """
    Integrates coordinates from a DataFrame into a single marker dictionary
    suitable for ``Animal3D('hawk', data=markers_dict)``.

    Parameters:
    - df: DataFrame with bird marker coordinates.
    - row_idx: Index of the row to extract.

    Returns:
    - markers_dict: Dictionary with all marker positions (moving + fixed).
    """
    markers = {}

    # Define marker mappings
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

