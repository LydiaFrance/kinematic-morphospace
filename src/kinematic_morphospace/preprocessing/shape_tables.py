"""
Construct unilateral and bilateral marker tables from long-format labelled data.

The unilateral table mirrors left-side markers (negating the X coordinate) and
pivots from long format (one row per marker per frame) to wide format (one row
per frame with columns for each marker type). Only pure-side frames (all 4
markers from the same side) are kept.

The bilateral table pivots all 8 markers (left + right x 4 types) into wide
format without mirroring.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Default marker types for unilateral pivoting
UNILATERAL_MARKERS = ["wingtip", "primary", "secondary", "tailtip"]

# Default marker names for bilateral pivoting (all 8 explicitly)
BILATERAL_MARKERS = [
    "left_wingtip", "right_wingtip",
    "left_primary", "right_primary",
    "left_secondary", "right_secondary",
    "left_tailtip", "right_tailtip",
]

# Metadata columns carried through the pivot
_INFO_COLS = [
    "frameID", "seqID", "time", "HorzDistance", "VertDistance",
    "body_pitch", "BirdID", "PerchDistance", "Year", "Naive",
    "Obstacle", "IMU",
]


# ---------------------------------------------------------------------------
# Shared pivot logic
# ---------------------------------------------------------------------------


def _get_coord_columns(df: pd.DataFrame, coord_prefix: str) -> list[str]:
    """Find the _1, _2, _3 columns for a given coordinate prefix."""
    return [f"{coord_prefix}_{i}" for i in (1, 2, 3) if f"{coord_prefix}_{i}" in df.columns]


def pivot_markers_wide(
    info_df: pd.DataFrame,
    markers_df: pd.DataFrame,
    markers: list[str],
    coord_prefix: str = "rot_xyz",
    use_contains: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Pivot marker data from long to wide format via successive inner joins.

    For each marker in *markers*, filters *markers_df* to matching rows,
    renames the coordinate columns to ``{marker}_{coord_prefix}_N``, and
    inner-joins with *info_df* on ``frameID``.

    Parameters
    ----------
    info_df : pd.DataFrame
        Deduplicated frame-level metadata (one row per frameID).
    markers_df : pd.DataFrame
        Long-format marker data with ``frameID``, ``MarkerName``, and
        coordinate columns (e.g. ``rot_xyz_1``, ``rot_xyz_2``, ``rot_xyz_3``).
    markers : list[str]
        Marker names/substrings to pivot (e.g. ``["wingtip", "primary"]``).
    coord_prefix : str
        Prefix of the coordinate columns (e.g. ``"rot_xyz"`` or ``"xyz"``).
    use_contains : bool
        If True, matches ``MarkerName`` using substring containment
        (``str.contains``). If False, uses exact matching (``isin``).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - The wide-format DataFrame with one row per frameID.
        - A list of all ``MarkerName`` columns accumulated during joins
          (for later left/right analysis).
    """
    coord_cols = _get_coord_columns(markers_df, coord_prefix)
    if not coord_cols:
        msg = f"No columns found with prefix '{coord_prefix}'"
        raise ValueError(msg)

    result = info_df.copy()
    marker_name_cols: list[str] = []

    for marker in markers:
        # Filter to rows matching this marker
        if use_contains:
            mask = markers_df["MarkerName"].str.contains(marker, na=False)
        else:
            mask = markers_df["MarkerName"] == marker
        subset = markers_df.loc[mask, ["frameID", "MarkerName", *coord_cols]].copy()

        if subset.empty:
            logger.warning("  No rows found for marker '%s'", marker)
            continue

        # Rename coordinate columns to include marker name
        rename_map = {
            c: f"{marker}_{c}" for c in coord_cols
        }
        marker_col_name = f"MarkerName_{marker}"
        rename_map["MarkerName"] = marker_col_name
        subset = subset.rename(columns=rename_map)

        marker_name_cols.append(marker_col_name)

        # Inner join — only frames with this marker survive
        result = result.merge(subset, on="frameID", how="inner")
        logger.info(
            "    Joined '%s': %d frames remain", marker, len(result)
        )

    return result, marker_name_cols


# ---------------------------------------------------------------------------
# Mirroring
# ---------------------------------------------------------------------------


def mirror_left_markers(
    df: pd.DataFrame,
    coord_prefix: str = "rot_xyz",
    marker_col: str = "MarkerName",
) -> pd.DataFrame:
    """Negate the X coordinate for left-side markers.

    Identifies left markers by checking if *marker_col* contains ``"left"``.
    Negates the ``_1`` (X) component of the coordinate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format marker data.
    coord_prefix : str
        Prefix of coordinate columns.
    marker_col : str
        Column containing marker names.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with mirrored left-side X coordinates.
    """
    df = df.copy()
    x_col = f"{coord_prefix}_1"

    if x_col not in df.columns:
        logger.warning("Column '%s' not found — skipping mirroring", x_col)
        return df

    left_mask = df[marker_col].str.contains("left", na=False)
    n_mirrored = left_mask.sum()
    df.loc[left_mask, x_col] = -df.loc[left_mask, x_col]

    logger.info("  Mirrored %d left-side marker rows (X negated)", n_mirrored)
    return df


# ---------------------------------------------------------------------------
# Pure-side filtering
# ---------------------------------------------------------------------------


def filter_pure_side_frames(
    df: pd.DataFrame,
    marker_name_cols: list[str],
    n_markers: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep only frames where all markers are from the same side.

    Counts how many of the marker name columns contain ``"left"`` for each
    row. Keeps rows where the count is 0 (all right) or *n_markers* (all
    left). Discards mixed-side frames.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format marker data with accumulated MarkerName columns.
    marker_name_cols : list[str]
        Names of the MarkerName columns from the pivot step.
    n_markers : int
        Expected number of markers per side.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        - Filtered DataFrame with mixed-side rows removed.
        - Boolean Series indicating left-side frames (True = left).
    """
    # Count "left" occurrences across marker name columns
    left_count = pd.Series(0, index=df.index)
    for col in marker_name_cols:
        if col in df.columns:
            left_count += df[col].str.contains("left", na=False).astype(int)

    all_right = left_count == 0
    all_left = left_count == n_markers
    pure_side = all_right | all_left

    n_mixed = (~pure_side).sum()
    if n_mixed > 0:
        logger.info("  Removed %d mixed-side frames", n_mixed)

    filtered = df.loc[pure_side].copy()
    is_left = all_left.loc[pure_side]

    return filtered, is_left


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_unilateral_table(
    labelled_df: pd.DataFrame,
    markers: list[str] | None = None,
    coord_prefix: str = "rot_xyz",
    info_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create the unilateral marker table from long-format labelled data.

    Steps:
    1. Remove rows with any NaN values.
    2. Filter to the 4 marker types.
    3. Add ``VertDistance`` from smooth backpack Z.
    4. Mirror left-side X coordinates.
    5. Pivot to wide format via inner joins.
    6. Filter out mixed-side frames.
    7. Add ``Left`` boolean column.
    8. Deduplicate.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Full labelled marker table (long format).
    markers : list[str], optional
        Marker type substrings to include. Defaults to
        ``["wingtip", "primary", "secondary", "tailtip"]``.
    coord_prefix : str
        Coordinate column prefix (``"rot_xyz"`` or ``"xyz"``).
    info_cols : list[str], optional
        Metadata columns to carry through. Defaults to :data:`_INFO_COLS`.

    Returns
    -------
    pd.DataFrame
        Wide-format unilateral marker table.
    """
    markers = markers or UNILATERAL_MARKERS
    info_columns = info_cols or _INFO_COLS

    logger.info("Creating unilateral table (coord=%s)", coord_prefix)

    df = labelled_df.copy()

    # Step 1: Remove rows with any NaN
    n_before = len(df)
    df = df.dropna()
    logger.info("  Dropped %d rows with NaN (%d remain)", n_before - len(df), len(df))

    # Step 2: Filter to marker types (substring match)
    marker_mask = df["MarkerName"].str.contains(
        "|".join(markers), na=False
    )
    df = df.loc[marker_mask].copy()
    logger.info("  Filtered to %d marker rows", len(df))

    # Step 3: Add VertDistance from smooth backpack Z
    if "backpack_smooth_XYZ_3" in df.columns:
        df["VertDistance"] = df["backpack_smooth_XYZ_3"]
    elif "VertDistance" not in df.columns:
        logger.warning("  No backpack_smooth_XYZ_3 — VertDistance will be NaN")
        df["VertDistance"] = float("nan")

    # Step 4: Mirror left-side markers
    df = mirror_left_markers(df, coord_prefix=coord_prefix)

    # Step 5: Build info and marker subsets
    available_info = [c for c in info_columns if c in df.columns]
    coord_cols = _get_coord_columns(df, coord_prefix)
    marker_data = df[["frameID", "MarkerName", *coord_cols]].copy()
    info_data = df[available_info].drop_duplicates(subset="frameID")

    # Step 6: Pivot wide
    wide, marker_name_cols = pivot_markers_wide(
        info_data, marker_data, markers,
        coord_prefix=coord_prefix,
        use_contains=True,
    )

    # Step 7: Filter pure-side frames
    wide, is_left = filter_pure_side_frames(
        wide, marker_name_cols, n_markers=len(markers)
    )

    # Step 8: Add Left column, remove MarkerName columns
    wide["Left"] = is_left.astype(int)
    wide = wide.drop(
        columns=[c for c in marker_name_cols if c in wide.columns],
        errors="ignore",
    )

    # Step 9: Deduplicate
    n_before = len(wide)
    wide = wide.drop_duplicates()
    if n_before > len(wide):
        logger.info("  Removed %d duplicate rows", n_before - len(wide))

    wide = wide.reset_index(drop=True)
    logger.info("  Unilateral table: %d rows", len(wide))
    return wide


def create_bilateral_table(
    labelled_df: pd.DataFrame,
    markers: list[str] | None = None,
    coord_prefix: str = "rot_xyz",
    info_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create the bilateral marker table from long-format labelled data.

    Same as :func:`create_unilateral_table` but uses all 8 markers
    (left + right x 4 types), no mirroring, and no side filtering.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Full labelled marker table (long format).
    markers : list[str], optional
        Exact marker names to include. Defaults to all 8 bilateral markers.
    coord_prefix : str
        Coordinate column prefix.
    info_cols : list[str], optional
        Metadata columns to carry through.

    Returns
    -------
    pd.DataFrame
        Wide-format bilateral marker table.
    """
    markers = markers or BILATERAL_MARKERS
    info_columns = info_cols or _INFO_COLS

    logger.info("Creating bilateral table (coord=%s)", coord_prefix)

    df = labelled_df.copy()

    # Step 1: Remove rows with any NaN
    n_before = len(df)
    df = df.dropna()
    logger.info("  Dropped %d rows with NaN (%d remain)", n_before - len(df), len(df))

    # Step 2: Filter to marker names (exact match for bilateral)
    marker_mask = df["MarkerName"].isin(markers)
    df = df.loc[marker_mask].copy()
    logger.info("  Filtered to %d marker rows", len(df))

    # Step 3: Add VertDistance
    if "backpack_smooth_XYZ_3" in df.columns:
        df["VertDistance"] = df["backpack_smooth_XYZ_3"]
    elif "VertDistance" not in df.columns:
        df["VertDistance"] = float("nan")

    # Step 4: Build info and marker subsets (no mirroring)
    available_info = [c for c in info_columns if c in df.columns]
    coord_cols = _get_coord_columns(df, coord_prefix)
    marker_data = df[["frameID", "MarkerName", *coord_cols]].copy()
    info_data = df[available_info].drop_duplicates(subset="frameID")

    # Step 5: Pivot wide (exact matching, not substring)
    wide, marker_name_cols = pivot_markers_wide(
        info_data, marker_data, markers,
        coord_prefix=coord_prefix,
        use_contains=False,
    )

    # Step 6: Remove MarkerName columns
    wide = wide.drop(
        columns=[c for c in marker_name_cols if c in wide.columns],
        errors="ignore",
    )

    # Step 7: Deduplicate
    n_before = len(wide)
    wide = wide.drop_duplicates()
    if n_before > len(wide):
        logger.info("  Removed %d duplicate rows", n_before - len(wide))

    wide = wide.reset_index(drop=True)
    logger.info("  Bilateral table: %d rows", len(wide))
    return wide
