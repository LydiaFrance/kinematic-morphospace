import numpy as np
import pandas as pd
import yaml

from .data_filtering import filter_by
from .data_loading import merge_frame_info

"""
Functions for scaling marker data and adding supplementary information.

- scale_data: Scales the marker data by total wingspan.
- unscale_data: Reverses wingspan normalisation.
- add_turn_info: Adds turn direction to the frame info dataframe.
- add_tailpack_data: Adds tailpack marker data to the markers DataFrame.
- rename_tailpack_data: Renames tailpack columns to standard naming.
"""


def scale_data(data_csv, wingspan_path: str):
    """
    Scale the data by the total wingspan.

    Parameters:
    - data_csv (pd.DataFrame): A DataFrame containing the data.
    - wingspan_path (str): The path to the wingspan file.

    Returns:
    - data_csv (pd.DataFrame): A DataFrame containing the scaled data.
    """
    marker_cols = data_csv.columns[data_csv.columns.str.contains('_x|_y|_z')]

    with open(wingspan_path) as file:
        total_wingspans = yaml.load(file, Loader=yaml.FullLoader)

    print("Loaded total wingspans from TotalWingspans.yml")

    for hawk in ["Drogon", "Rhaegal", "Ruby", "Toothless", "Charmander"]:
        for year in [2017, 2020]:
            filter_mask = filter_by(data_csv, hawkname=hawk, year=year)

            if sum(filter_mask) == 0:
                continue

            wingspan = total_wingspans[hawk][year]
            data_csv.loc[filter_mask, marker_cols] = (
                data_csv.loc[filter_mask, marker_cols] / wingspan
            )

        print(f"-- {hawk} max wingspan estimated at {total_wingspans[hawk]}")

    print("Scaled the data.")

    return data_csv


def unscale_data(data_csv, wingspan_path: str):
    """
    Reverse the wingspan normalisation.

    Parameters:
    - data_csv (pd.DataFrame): A DataFrame containing the data.
    - wingspan_path (str): The path to the wingspan file.

    Returns:
    - data_csv (pd.DataFrame): A DataFrame containing the unscaled data.
    """
    marker_cols = data_csv.columns[data_csv.columns.str.contains('_x|_y|_z')]

    with open(wingspan_path) as file:
        total_wingspans = yaml.load(file, Loader=yaml.FullLoader)

    print("Loaded total wingspans from TotalWingspans.yml")

    for hawk in ["Drogon", "Rhaegal", "Ruby", "Toothless", "Charmander"]:
        for year in [2017, 2020]:
            filter_mask = filter_by(data_csv, hawkname=hawk, year=year)

            if sum(filter_mask) == 0:
                continue

            wingspan = total_wingspans[hawk][year]
            data_csv.loc[filter_mask, marker_cols] = (
                data_csv.loc[filter_mask, marker_cols] * wingspan
            )

        print(f"-- {hawk} max wingspan estimated at {total_wingspans[hawk]}")

    print("Unscaled the data.")

    return data_csv


def add_turn_info(frame_info_df, turn_csv_path: str):
    """
    Add turn information to the frame info dataframe.
    Turn information is the direction the hawk flew around the obstacle.

    Parameters:
    - frame_info_df (pd.DataFrame): A DataFrame containing the frame info.
    - turn_csv_path (str): The path to the turn CSV file.

    Returns:
    - frame_info_df (pd.DataFrame): A DataFrame with turn information added.
    """
    frame_info_df = frame_info_df.copy()

    obstacle_df = pd.read_csv(turn_csv_path)

    # Remove any existing turn columns
    frame_info_df = frame_info_df.drop(
        columns=['Turn', 'Turn_x', 'Turn_y'], errors='ignore'
    )

    if 'Turn' not in frame_info_df.columns:
        frame_info_df = frame_info_df.merge(obstacle_df, how='left', on='seqID')

    print("Added turn direction to dataframe.")

    return frame_info_df


def add_tailpack_data(markers_df, frame_info_df, tailpack_csv, wingspan_path=None):
    """
    Add tailpack data to the markers DataFrame.

    Parameters:
    - markers_df (pd.DataFrame): A DataFrame containing the markers.
    - frame_info_df (pd.DataFrame): A DataFrame containing the frame info.
    - tailpack_csv (str): The path to the tailpack CSV file.
    - wingspan_path (str, optional): Path to TotalWingspans.yml. If provided,
      tailpack coordinates are scaled by wingspan to match the wing markers.

    Returns:
    - combined_markers (numpy.ndarray): Markers including tailpack data.
    - combined_frame_info_df (pd.DataFrame): Frame info aligned with tailpack data.
    """
    n_markers_original = len(markers_df.columns) // 3
    print(f"Original number of markers: {n_markers_original}")

    tailpack_df = pd.read_csv(tailpack_csv)
    tailpack_df = rename_tailpack_data(tailpack_df)

    # Merge the tailpack data, returning an index of non-NaN rows
    merged_df, row_index = merge_frame_info(tailpack_df, frame_info_df)

    # Use NaN index to remove rows without tailpack data
    combined_frame_info_df = frame_info_df.copy()
    combined_frame_info_df = combined_frame_info_df[row_index].reset_index(drop=True)

    # Build the markers numpy array
    marker_cols = merged_df.columns[merged_df.columns.str.contains('_x|_y|_z')]

    # Scale tailpack by wingspan if wingspan_path is provided
    if wingspan_path is not None:
        with open(wingspan_path) as file:
            total_wingspans = yaml.load(file, Loader=yaml.FullLoader)

        for hawk in ["Drogon", "Rhaegal", "Ruby", "Toothless", "Charmander"]:
            for year in [2017, 2020]:
                # merged_df has frame_info columns from the merge, so we can
                # filter directly on it rather than needing combined_frame_info_df
                filter_mask = filter_by(merged_df, hawkname=hawk, year=year)
                if sum(filter_mask) == 0:
                    continue
                wingspan = total_wingspans[hawk][year]
                merged_df.loc[filter_mask, marker_cols] = (
                    merged_df.loc[filter_mask, marker_cols] / wingspan
                )

        print("Scaled tailpack data by wingspan.")

    merged_df = merged_df[marker_cols]
    n_markers = len(marker_cols) // 3

    combined_markers = merged_df.to_numpy().reshape(-1, n_markers, 3)

    # Align the original markers to the same row subset
    markers_df_new = markers_df.copy()
    markers_df_new = markers_df_new[row_index].reset_index(drop=True)

    markers = markers_df_new.to_numpy().reshape(-1, n_markers_original, 3)
    combined_markers = np.concatenate((markers, combined_markers), axis=1)
    new_n_markers = len(combined_markers[0])

    if new_n_markers != n_markers_original + 1:
        print(f"Error: Expected {n_markers_original + 1} markers, got {new_n_markers}.")

    return combined_markers, combined_frame_info_df


def rename_tailpack_data(markers_df):
    """
    Rename tailpack columns to standard marker naming convention.
    """
    markers_df = markers_df.copy()

    rot_markers_cols = ["rot_xyz_1", "rot_xyz_2", "rot_xyz_3"]
    new_name_cols = ["tailpack_x", "tailpack_y", "tailpack_z"]

    markers_df.rename(
        columns=dict(zip(rot_markers_cols, new_name_cols)), inplace=True
    )

    remove_cols = ["xyz_1", "xyz_2", "xyz_3"]
    markers_df.drop(columns=remove_cols, errors='ignore', inplace=True)

    return markers_df
