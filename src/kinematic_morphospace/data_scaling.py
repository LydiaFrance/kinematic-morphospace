"""Functions for scaling marker data and adding supplementary information.

Provides wingspan normalisation, its inverse, turn-direction annotation,
and tailpack marker integration.
"""

import numpy as np
import pandas as pd
import yaml

from .data_filtering import filter_by
from .data_loading import merge_frame_info


def scale_data(data_csv, wingspan_path: str):
    """Normalise marker coordinates by each hawk's maximum total wingspan.

    Divides all x/y/z marker columns by the per-hawk, per-year maximum
    wingspan loaded from a YAML file, so that shapes are expressed as
    fractions of wingspan and are comparable across individuals and years.

    Args:
        data_csv: DataFrame containing marker coordinate columns (identified
            by ``_x``, ``_y``, ``_z`` suffixes) and ``BirdID`` / ``Year``
            metadata columns.
        wingspan_path: Path to ``TotalWingspans.yml``, which maps hawk name
            and year to the maximum recorded wingspan in metres.

    Returns:
        Copy of ``data_csv`` with marker columns divided by wingspan.
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
    """Reverse the wingspan normalisation applied by ``scale_data``.

    Multiplies all x/y/z marker columns by the per-hawk, per-year maximum
    wingspan, restoring coordinates to metres.

    Args:
        data_csv: DataFrame containing normalised marker coordinate columns
            and ``BirdID`` / ``Year`` metadata columns.
        wingspan_path: Path to ``TotalWingspans.yml``.

    Returns:
        Copy of ``data_csv`` with marker columns multiplied by wingspan.
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
    """Add turn-direction annotation to the frame-info DataFrame.

    Merges per-sequence turn direction (left, right, or straight around
    an obstacle) from a CSV file into the frame metadata. Turn information
    is required for obstacle-flight analyses that distinguish approach
    trajectories.

    Args:
        frame_info_df: DataFrame containing per-frame metadata with a
            ``seqID`` column for joining.
        turn_csv_path: Path to the CSV file containing ``seqID`` and
            ``Turn`` columns.

    Returns:
        Copy of ``frame_info_df`` with a ``Turn`` column added.
    """
    frame_info_df = frame_info_df.copy()

    obstacle_df = pd.read_csv(turn_csv_path)

    frame_info_df = frame_info_df.drop(
        columns=['Turn', 'Turn_x', 'Turn_y'], errors='ignore'
    )

    if 'Turn' not in frame_info_df.columns:
        frame_info_df = frame_info_df.merge(obstacle_df, how='left', on='seqID')

    print("Added turn direction to dataframe.")

    return frame_info_df


def add_tailpack_data(markers_df, frame_info_df, tailpack_csv, wingspan_path=None):
    """Append tailpack marker coordinates to the markers DataFrame.

    Loads a tailpack CSV, aligns it with the existing frame metadata via
    ``frameID``, optionally normalises by wingspan, and concatenates the
    tailpack marker with the existing wing markers. Frames without tailpack
    data are dropped, and the frame-info DataFrame is trimmed to match.

    Args:
        markers_df: DataFrame of wing marker coordinates (one marker trio
            per group of three columns).
        frame_info_df: DataFrame of per-frame metadata aligned with
            ``markers_df``.
        tailpack_csv: Path to the tailpack marker CSV file.
        wingspan_path: Path to ``TotalWingspans.yml``. If provided,
            tailpack coordinates are normalised by wingspan to match the
            already-scaled wing markers. Defaults to None.

    Returns:
        Tuple of (combined_markers, combined_frame_info_df) where
        combined_markers has shape ``(n_frames, n_wing_markers + 1, 3)``
        and combined_frame_info_df is the trimmed frame metadata.
    """
    n_markers_original = len(markers_df.columns) // 3
    print(f"Original number of markers: {n_markers_original}")

    tailpack_df = pd.read_csv(tailpack_csv)
    tailpack_df = rename_tailpack_data(tailpack_df)

    merged_df, row_index = merge_frame_info(tailpack_df, frame_info_df)

    combined_frame_info_df = frame_info_df.copy()
    combined_frame_info_df = combined_frame_info_df[row_index].reset_index(drop=True)

    marker_cols = merged_df.columns[merged_df.columns.str.contains('_x|_y|_z')]

    if wingspan_path is not None:
        with open(wingspan_path) as file:
            total_wingspans = yaml.load(file, Loader=yaml.FullLoader)

        for hawk in ["Drogon", "Rhaegal", "Ruby", "Toothless", "Charmander"]:
            for year in [2017, 2020]:
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

    combined_markers = np.asarray(merged_df).reshape(-1, n_markers, 3)

    markers_df_new = markers_df.copy()
    markers_df_new = markers_df_new[row_index].reset_index(drop=True)

    markers = markers_df_new.to_numpy().reshape(-1, n_markers_original, 3)
    combined_markers = np.concatenate((markers, combined_markers), axis=1)
    new_n_markers = len(combined_markers[0])

    if new_n_markers != n_markers_original + 1:
        print(f"Error: Expected {n_markers_original + 1} markers, got {new_n_markers}.")

    return combined_markers, combined_frame_info_df


def rename_tailpack_data(markers_df):
    """Rename tailpack columns to the standard marker naming convention.

    Maps the raw ``rot_xyz_1/2/3`` column names produced by the motion
    capture pipeline to ``tailpack_x/y/z`` and removes any raw ``xyz``
    columns that are not needed.

    Args:
        markers_df: DataFrame containing raw tailpack marker columns.

    Returns:
        Copy of ``markers_df`` with renamed columns.
    """
    markers_df = markers_df.copy()

    rot_markers_cols = ["rot_xyz_1", "rot_xyz_2", "rot_xyz_3"]
    new_name_cols = ["tailpack_x", "tailpack_y", "tailpack_z"]

    markers_df.rename(
        columns=dict(zip(rot_markers_cols, new_name_cols, strict=False)), inplace=True
    )

    remove_cols = ["xyz_1", "xyz_2", "xyz_3"]
    markers_df.drop(columns=remove_cols, errors='ignore', inplace=True)

    return markers_df
