"""Harmonise raw MATLAB-derived DataFrames into a consistent schema.

Handles column renaming, metadata enrichment (BirdID, Year, Obstacle, IMU,
Naive, PerchDistance), sequence ID extraction, and outer joins for body_pitch
and smooth_XYZ data.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bird metadata constants (defaults matching the MATLAB script)
# ---------------------------------------------------------------------------

#: Birds that are NOT naive in the 2017 campaign (BirdID -> is_naive).
NAIVE_EXCEPTIONS_2017: dict[int, bool] = {3: False}  # Ruby

#: Birds that ARE naive in the 2020 campaign (BirdID -> is_naive).
NAIVE_EXCEPTIONS_2020: dict[int, bool] = {5: True}  # Charmander


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

# Standard column order for the leading metadata columns (shared by
# trajectory and labelled output tables).
_META_COLS = [
    "frameID", "seqID", "Year", "Obstacle", "IMU", "Naive",
    "BirdID", "PerchDistance",
]


def _reorder_columns(
    df: pd.DataFrame,
    leading: list[str],
) -> pd.DataFrame:
    """Move *leading* columns to the front, keeping the rest in order."""
    present = [c for c in leading if c in df.columns]
    rest = [c for c in df.columns if c not in present]
    return df[present + rest]


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------


def extract_bird_id(frame_ids: pd.Series) -> pd.Series:
    """Extract numeric BirdID from the second character of each frameID string.

    In the hawk dataset, frameIDs follow the pattern ``D3_seq1_001``
    where the digit after the first letter is the bird number.

    Args:
        frame_ids: Series of frameID strings in the dataset's naming
            convention.

    Returns:
        Integer Series of BirdID values.
    """
    return frame_ids.str[1].astype(int)


def extract_seq_id(frame_ids: pd.Series) -> pd.Series:
    """Derive seqID by stripping the trailing frame-number segment from frameID.

    For example, ``"D3_5m_seq4_042"`` becomes ``"D3_5m_seq4"``.

    Args:
        frame_ids: Series of frameID strings in the dataset's naming
            convention.

    Returns:
        Series of sequence ID strings with the ``_frameNum`` suffix removed.
    """
    return frame_ids.str.rsplit("_", n=1).str[0]


def add_metadata(
    df: pd.DataFrame,
    year: int,
    info_df: pd.DataFrame | None = None,
    *,
    naive_exceptions: dict[int, bool] | None = None,
    default_perch_distance: float | None = None,
) -> pd.DataFrame:
    """Add standard experimental metadata columns to a DataFrame.

    Adds or updates ``BirdID``, ``Year``, ``Obstacle``, ``IMU``, ``Naive``,
    and ``PerchDistance``. Per-campaign defaults are used for naive status
    and perch distance unless overridden.

    Args:
        df: DataFrame that must contain a ``frameID`` column (and ``seqID``
            for 2020 obstacle/IMU lookup).
        year: Data collection campaign year, either 2017 or 2020.
        info_df: The ``asymInfo`` table (2020 only) containing ``seqID``,
            ``Obstacle``, and ``IMU`` columns. Ignored for 2017.
        naive_exceptions: Per-bird overrides for naive status as a dict
            mapping BirdID (int) to bool. Defaults to campaign-appropriate
            values from :data:`NAIVE_EXCEPTIONS_2017` or
            :data:`NAIVE_EXCEPTIONS_2020`.
        default_perch_distance: Default perch separation in metres for all
            rows. Defaults to 9.0 for 2020; the 2017 table is expected to
            already carry a ``PerchDistance`` column.

    Returns:
        Copy of ``df`` with the metadata columns added or updated.
    """
    df = df.copy()

    # BirdID — extract from frameID if not already present
    if "BirdID" not in df.columns:
        df["BirdID"] = extract_bird_id(df["frameID"])

    # Year
    df["Year"] = year

    # PerchDistance — 2020 hardcoded to 9, 2017 already has variable values
    if "PerchDistance" not in df.columns:
        perch = default_perch_distance if default_perch_distance is not None else 9.0
        df["PerchDistance"] = perch

    # Obstacle and IMU
    if year == 2020 and info_df is not None:
        df = _lookup_info_flags(df, info_df)
    else:
        if "Obstacle" not in df.columns:
            df["Obstacle"] = 0
        if "IMU" not in df.columns:
            df["IMU"] = 0

    # Naive
    exceptions = naive_exceptions
    if exceptions is None:
        exceptions = (
            NAIVE_EXCEPTIONS_2017 if year == 2017 else NAIVE_EXCEPTIONS_2020
        )
    default_naive = 1 if year == 2017 else 0
    df["Naive"] = default_naive
    for bird_id, is_naive in exceptions.items():
        df.loc[df["BirdID"] == bird_id, "Naive"] = int(is_naive)

    return df


def _lookup_info_flags(
    df: pd.DataFrame,
    info_df: pd.DataFrame,
) -> pd.DataFrame:
    """Set Obstacle and IMU flags by looking up seqID in the info table.

    Uses substring matching (``str.contains``) to replicate the MATLAB
    ``contains()`` behaviour.
    """
    df["Obstacle"] = 0
    df["IMU"] = 0

    if "seqID" not in df.columns:
        logger.warning("No seqID column — cannot look up Obstacle/IMU flags")
        return df

    obstacle_seqs = info_df.loc[info_df["Obstacle"] == 1, "seqID"].tolist()
    for seq in obstacle_seqs:
        mask = df["seqID"].str.contains(seq, na=False)
        df.loc[mask, "Obstacle"] = 1

    imu_seqs = info_df.loc[info_df["IMU"] == 1, "seqID"].tolist()
    for seq in imu_seqs:
        mask = df["seqID"].str.contains(seq, na=False)
        df.loc[mask, "IMU"] = 1

    return df


# ---------------------------------------------------------------------------
# Trajectory harmonisation
# ---------------------------------------------------------------------------


def harmonise_trajectory(
    traj_df: pd.DataFrame,
    year: int,
    info_df: pd.DataFrame | None = None,
    tail_df: pd.DataFrame | None = None,
    smooth_df: pd.DataFrame | None = None,
    *,
    label: str = "mean_backpack",
    drop_cols_2017: list[str] | None = None,
) -> pd.DataFrame:
    """Harmonise a raw trajectory DataFrame into the shared analysis schema.

    Applies column renaming, label assignment (2020), metadata enrichment,
    body pitch joining, smooth XYZ joining (2017 only), column removal,
    and consistent column ordering. This step reconciles the structural
    differences between the 2017 and 2020 MATLAB outputs.

    Args:
        traj_df: Raw trajectory table as loaded by
            :func:`mat_loader.load_2020_data` or
            :func:`mat_loader.load_2017_data`.
        year: Campaign year (2017 or 2020).
        info_df: The ``asymInfo`` table (2020 only) used to set Obstacle and
            IMU flags.
        tail_df: Tail/tailpack table containing ``frameID`` and
            ``body_pitch`` for joining.
        smooth_df: Smooth body table with ``XYZ_*`` columns for joining
            (2017 only).
        label: Label string applied to all rows in the 2020 trajectory table.
            Defaults to ``"mean_backpack"``.
        drop_cols_2017: Columns to remove from the 2017 table. Defaults to
            ``["OriginalSequence", "sampleRate", "TotalForce", "SCLift",
            "SCDrag", "climb_angle"]``.

    Returns:
        Harmonised trajectory DataFrame with consistent schema across both
        campaign years.
    """
    df = traj_df.copy()
    logger.info("Harmonising %d trajectory table (%d rows)", year, len(df))

    # --- Set label for 2020 ---
    if year == 2020:
        df["label"] = label

    # --- Metadata ---
    df = add_metadata(df, year, info_df)

    # --- Join body_pitch from tail table ---
    if tail_df is not None:
        df = join_body_pitch(df, tail_df)

    # --- Join smooth_XYZ (2017 only) ---
    if smooth_df is not None and year == 2017:
        df = join_smooth_xyz(df, smooth_df)

    # --- Drop unused columns ---
    if year == 2017:
        cols_to_drop = drop_cols_2017 or [
            "OriginalSequence", "sampleRate", "TotalForce",
            "SCLift", "SCDrag", "climb_angle",
        ]
        df = df.drop(
            columns=[c for c in cols_to_drop if c in df.columns],
            errors="ignore",
        )

    # --- Add NaN mass for 2020 ---
    if year == 2020 and "mass" not in df.columns:
        df["mass"] = np.nan

    # --- Reorder ---
    df = _reorder_columns(df, _META_COLS)

    return df


# ---------------------------------------------------------------------------
# Labelled table harmonisation
# ---------------------------------------------------------------------------


def harmonise_labelled(
    labelled_df: pd.DataFrame,
    year: int,
    info_df: pd.DataFrame | None = None,
    *,
    drop_cols_2017: list[str] | None = None,
    drop_cols_2020: list[str] | None = None,
) -> pd.DataFrame:
    """Harmonise a raw labelled-marker DataFrame into the shared analysis schema.

    Derives ``seqID`` for 2017, enriches with experimental metadata, drops
    campaign-specific columns, and reorders to the standard column layout.

    Args:
        labelled_df: Raw labelled marker table from MATLAB or CSV loading.
        year: Campaign year (2017 or 2020).
        info_df: The ``asymInfo`` table (2020 only) used to set Obstacle and
            IMU flags.
        drop_cols_2017: Columns to remove from the 2017 table. Defaults to
            ``["OriginalSequence", "sampleRate", "markerID"]``.
        drop_cols_2020: Columns to remove from the 2020 table. Defaults to
            ``["label_Vicon", "label_stationary", "ID", "trial",
            "markerID"]``.

    Returns:
        Harmonised labelled marker DataFrame with consistent schema across
        both campaign years.
    """
    df = labelled_df.copy()
    logger.info("Harmonising %d labelled table (%d rows)", year, len(df))

    # --- seqID for 2017 (derived from frameID) ---
    if year == 2017 and "seqID" not in df.columns:
        df["seqID"] = extract_seq_id(df["frameID"])

    # --- Metadata ---
    df = add_metadata(df, year, info_df)

    # --- Drop unused columns ---
    if year == 2017:
        cols_to_drop = drop_cols_2017 or [
            "OriginalSequence", "sampleRate", "markerID",
        ]
    else:
        cols_to_drop = drop_cols_2020 or [
            "label_Vicon", "label_stationary", "ID", "trial", "markerID",
        ]
    df = df.drop(
        columns=[c for c in cols_to_drop if c in df.columns],
        errors="ignore",
    )

    # --- Reorder ---
    df = _reorder_columns(df, _META_COLS)

    return df


# ---------------------------------------------------------------------------
# Join helpers
# ---------------------------------------------------------------------------


def join_body_pitch(
    df: pd.DataFrame,
    tail_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join body pitch values from the tail table onto the main DataFrame.

    Matches by ``frameID``; frames in ``df`` without a corresponding entry in
    ``tail_df`` receive NaN. If ``body_pitch`` already exists in ``df``, the
    tail-derived values take precedence.

    Args:
        df: Main DataFrame (trajectory or labelled marker table) to enrich
            with pitch data.
        tail_df: Tail/tailpack table containing ``frameID`` and
            ``body_pitch`` columns.

    Returns:
        ``df`` with a ``body_pitch`` column added (NaN where no match).
    """
    if "body_pitch" not in tail_df.columns:
        logger.warning("tail_df has no body_pitch column — skipping join")
        return df

    pitch = tail_df[["frameID", "body_pitch"]].copy()
    pitch = pitch.drop_duplicates(subset="frameID")

    merged = df.merge(pitch, on="frameID", how="left", suffixes=("", "_tail"))

    # If body_pitch already existed, prefer the tail version
    if "body_pitch_tail" in merged.columns:
        merged["body_pitch"] = merged["body_pitch_tail"].combine_first(
            merged["body_pitch"]
        )
        merged = merged.drop(columns=["body_pitch_tail"])

    logger.info("  Joined body_pitch: %d matched", merged["body_pitch"].notna().sum())
    return merged


def join_smooth_xyz(
    df: pd.DataFrame,
    smooth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join smoothed XYZ position columns from the smooth body table.

    Renames the smooth table's ``XYZ_*`` columns to ``smooth_XYZ_*`` to
    avoid collision with raw coordinates, then merges by ``frameID``.

    Args:
        df: Main trajectory table to enrich with smoothed positions.
        smooth_df: Smooth body table with ``frameID`` and
            ``XYZ_1``/``XYZ_2``/``XYZ_3`` columns.

    Returns:
        ``df`` with ``smooth_XYZ_1``, ``smooth_XYZ_2``, ``smooth_XYZ_3``
        columns added (NaN where no match).
    """
    xyz_cols = [c for c in smooth_df.columns if c.startswith("XYZ_")]
    if not xyz_cols:
        logger.warning("smooth_df has no XYZ columns — skipping join")
        return df

    smooth_subset = smooth_df[["frameID", *xyz_cols]].copy()
    smooth_subset = smooth_subset.drop_duplicates(subset="frameID")

    # Rename XYZ_* -> smooth_XYZ_* to avoid collision
    rename_map = {c: f"smooth_{c}" for c in xyz_cols}
    smooth_subset = smooth_subset.rename(columns=rename_map)

    merged = df.merge(smooth_subset, on="frameID", how="left")
    n_matched = merged[f"smooth_{xyz_cols[0]}"].notna().sum()
    logger.info("  Joined smooth_XYZ: %d matched", n_matched)
    return merged
