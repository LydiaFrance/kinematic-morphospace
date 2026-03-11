"""Functions for loading and processing motion-capture data.

Provides utilities for reading CSV marker data, filtering frames by
spatial and temporal limits, and converting DataFrames into NumPy arrays
suitable for PCA analysis.
"""

import numpy as np
import pandas as pd


# ------- Loading data -------

def load_data(csv_path: str):
    """Load and return data from a CSV file.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file containing the data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the loaded data.
    """
    data_csv = pd.read_csv(csv_path)

    return data_csv


def remove_frames(data_csv, Y_limit=0.1, time_limit=0):
    """Remove frames outside valid spatial and temporal limits.

    The default *Y_limit* is 0.1 m.  Y has the origin at the destination
    perch, so values are negative unless the recordings are post-landing.
    Frames beyond this point are removed.  The default *time_limit* is 0,
    which removes pre-takeoff frames.

    Parameters
    ----------
    data_csv : pd.DataFrame
        A DataFrame containing the motion-capture data.
    Y_limit : float, optional
        Maximum allowed Y-axis (or horizontal distance) value.
    time_limit : float, optional
        Minimum allowed time value.

    Returns
    -------
    pd.DataFrame
        A DataFrame with invalid frames removed.
    """
    print(f"Starting with {len(data_csv)} frames.")

    data_csv = data_csv.copy()

    # Remove frames where the time is less than the given limit
    print(f"Removed frames with time less than {time_limit}. "
          "This removes frames before takeoff jump.")
    frames_removed = sum(data_csv['time'] < time_limit)
    print(f"-- Number of frames removed: {frames_removed}")
    data_csv = data_csv[data_csv['time'] > time_limit].reset_index(drop=True)

    # Determine the nature of HorzDistance values (mostly negative or positive)
    if sum(data_csv['HorzDistance'] < 0) > len(data_csv) / 2:
        print("Detected mostly negative HorzDistance values.")
    else:
        print("Detected mostly positive HorzDistance values. Flipping the sign.")
        data_csv['HorzDistance'] *= -1

    # Remove frames past the perch
    if 'smooth_XYZ_2' in data_csv.columns:
        frames_removed = sum(data_csv['smooth_XYZ_2'] > Y_limit)
        print(f"Removed frames with Y greater than {Y_limit}.")
        data_csv = data_csv[data_csv['smooth_XYZ_2'] < Y_limit].reset_index(drop=True)
    else:
        frames_removed = sum(data_csv['HorzDistance'] > Y_limit)
        print(f"Removed frames with horizontal distance greater than {Y_limit}.")
        data_csv = data_csv[data_csv['HorzDistance'] < Y_limit].reset_index(drop=True)

    print(f"-- Number of frames removed: {frames_removed}")
    print(f"Now {len(data_csv)} frames.")

    return data_csv


def prepare_marker_data(df, n_markers=8):
    """Convert marker coordinate columns into a 3-D NumPy array.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing ``xyz_1``, ``xyz_2``, ``xyz_3`` columns.
    n_markers : int, optional
        The number of markers per frame.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_frames, n_markers, 3)``.
    """
    x_coords = df['xyz_1'].values.reshape(-1, n_markers)
    y_coords = df['xyz_2'].values.reshape(-1, n_markers)
    z_coords = df['xyz_3'].values.reshape(-1, n_markers)

    return np.stack([x_coords, y_coords, z_coords], axis=2)


def process_data(data_csv):
    """Generate NumPy arrays and DataFrames from loaded motion-capture CSV data.

    Splits the CSV into marker coordinates and frame metadata, converts
    them to arrays, and verifies that all lengths match.

    Parameters
    ----------
    data_csv : pd.DataFrame
        Loaded CSV containing marker information and frame metadata.

    Returns
    -------
    markers : np.ndarray
        Marker coordinate array of shape ``(n_frames, n_markers, 3)``.
    frame_info : dict
        Dictionary of per-frame metadata arrays.
    markers_df : pd.DataFrame
        DataFrame of cleaned marker columns.
    frame_info_df : pd.DataFrame
        DataFrame of non-marker frame metadata.
    """
    markers_df = load_marker_frames(data_csv.copy())
    frame_info_df = load_frame_info(data_csv.copy())

    markers, frame_info = get_arrays(markers_df, frame_info_df)

    if check_data(markers, frame_info):
        print("Data verified.")
    else:
        print("Data verification failed.")

    return markers, frame_info, markers_df, frame_info_df


# ------- Helper functions -------

def load_marker_frames(markers_csv):
    """Extract and rename marker coordinate columns from the CSV.

    Keeps only columns containing ``rot_xyz`` and renames them to
    ``<marker>_x``, ``<marker>_y``, ``<marker>_z``.

    Parameters
    ----------
    markers_csv : pd.DataFrame
        Raw CSV DataFrame with ``rot_xyz`` marker columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the renamed marker columns.
    """
    markers_csv.drop(
        columns=markers_csv.columns[~markers_csv.columns.str.contains('rot_xyz')],
        inplace=True,
    )

    markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_1", "_x")
    markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_2", "_y")
    markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_3", "_z")

    print("Cleaned marker names.")

    return markers_csv


def load_frame_info(frame_info_csv):
    """Return frame metadata by dropping marker coordinate columns.

    Removes all columns containing ``rot_xyz``, leaving only non-marker
    metadata (e.g. time, distance, bird ID).

    Parameters
    ----------
    frame_info_csv : pd.DataFrame
        Raw CSV DataFrame including both marker and metadata columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with marker columns removed.
    """
    frame_info_csv.drop(
        columns=frame_info_csv.columns[frame_info_csv.columns.str.contains('rot_xyz')],
        inplace=True,
    )

    print("Loaded frame info and cleaned columns.")

    return frame_info_csv


def get_arrays(markers_csv, frame_info_csv):
    """Convert marker and frame-info DataFrames into NumPy arrays.

    Parameters
    ----------
    markers_csv : pd.DataFrame
        DataFrame whose columns follow the ``<marker>_x/_y/_z`` pattern.
    frame_info_csv : pd.DataFrame
        DataFrame of non-marker frame metadata.

    Returns
    -------
    markers : np.ndarray
        Array of shape ``(n_frames, n_markers, 3)``.
    frame_info : dict
        Dictionary mapping metadata keys to NumPy arrays or Series.
    """
    marker_cols = markers_csv.columns[markers_csv.columns.str.contains('_x|_y|_z')]
    markers_csv = markers_csv[marker_cols]
    n_markers = len(marker_cols) // 3

    markers = markers_csv.to_numpy().reshape(-1, n_markers, 3)

    frame_info = {
        'time':         get_column_as_numpy(frame_info_csv, 'time'),
        'horzDist':     get_column_as_numpy(frame_info_csv, 'HorzDistance'),
        'vertDist':     get_column_as_numpy(frame_info_csv, 'VertDistance'),
        'body_pitch':   get_column_as_numpy(frame_info_csv, 'body_pitch'),
        'frameID':      get_column(frame_info_csv, 'frameID'),
        'seqID':        get_column(frame_info_csv, 'seqID'),
        'birdID':       get_column_as_numpy(frame_info_csv, 'BirdID'),
        'perchDist':    get_column_as_numpy(frame_info_csv, 'PerchDistance'),
        'year':         get_column_as_numpy(frame_info_csv, 'Year'),
        'obstacleBool': get_column_as_numpy(frame_info_csv, 'Obstacle'),
        'IMUBool':      get_column_as_numpy(frame_info_csv, 'IMU'),
        'naiveBool':    get_column_as_numpy(frame_info_csv, 'Naive'),
        'leftBool':     get_column(frame_info_csv, 'Left'),
    }

    return markers, frame_info


def check_data(markers, frame_info):
    """Verify that all arrays in *frame_info* and *markers* share the same length.

    Parameters
    ----------
    markers : np.ndarray
        Marker coordinate array.
    frame_info : dict
        Dictionary of per-frame metadata arrays.

    Returns
    -------
    bool
        ``True`` if every array has the same first-axis length,
        ``False`` otherwise.
    """
    all_data = list(frame_info.values()) + [markers]
    array_lengths = [len(data) for data in all_data if isinstance(data, np.ndarray)]

    if len(set(array_lengths)) == 1:
        return True
    else:
        mismatch_info = {
            key: len(value) for key, value in frame_info.items()
            if isinstance(value, np.ndarray)
        }
        mismatch_info['markers'] = len(markers)
        print("Mismatch in data lengths found:", mismatch_info)
        return False


def get_column_as_numpy(df, column_name):
    """Retrieve a DataFrame column as a NumPy array, or ``None`` if absent.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    column_name : str
        Name of the column to retrieve.

    Returns
    -------
    np.ndarray or None
        Column values as a NumPy array, or ``None`` if the column does
        not exist.
    """
    return df.get(column_name).to_numpy() if column_name in df.columns else None


def get_column(df, column_name):
    """Retrieve a DataFrame column as a Series, or ``None`` if absent.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    column_name : str
        Name of the column to retrieve.

    Returns
    -------
    pd.Series or None
        Column as a Series, or ``None`` if the column does not exist.
    """
    return df.get(column_name) if column_name in df.columns else None


def merge_frame_info(df, frame_info_df):
    """Merge supplementary data with the frame-info DataFrame.

    Performs a left join on ``frameID``, removes rows containing NaN
    values, and returns a boolean index indicating which rows from
    *frame_info_df* are present in the merged result.

    Parameters
    ----------
    df : pd.DataFrame
        Supplementary DataFrame that must contain a ``frameID`` column.
    frame_info_df : pd.DataFrame
        Frame metadata DataFrame with a ``frameID`` column.

    Returns
    -------
    df_tail : pd.DataFrame
        Merged DataFrame with NaN rows removed.
    row_index : pd.Series
        Boolean index of rows in *frame_info_df* present in the result.
    """
    print(f"Merging frame info with markers DataFrame. "
          f"Frame info: {frame_info_df.shape}, New df: {df.shape}")

    df_tail = df.copy()
    df_tail = df_tail.merge(frame_info_df, how='left', on='frameID')

    # Remove rows with NaN values
    isnan_index = df_tail.isna().any(axis=1)

    print(f"Removed {sum(isnan_index)} rows out of {df_tail.shape} with NaN values.")
    df_tail = df_tail[~isnan_index].reset_index(drop=True)

    # Produce a boolean index of rows in frame_info_df present in the merged result
    row_index = frame_info_df['frameID'].isin(df_tail['frameID'])
    print(f"Rows in frame_info_df not in merged df: {sum(~row_index)}")

    return df_tail, row_index
