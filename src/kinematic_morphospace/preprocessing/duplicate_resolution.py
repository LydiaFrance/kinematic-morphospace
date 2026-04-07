"""Duplicate marker detection and resolution for wing labelling.

After polygon-based labelling, multiple markers in the same frame may receive
the same label. This module detects such duplicates and resolves them using
distance-based heuristics that mirror the logic in ``run_wing_labelling.m``
(lines 584-733).

Three public functions:

- :func:`detect_duplicates` — partition rows into unique, duplicate pairs, and
  excess (3+ with the same label in a frame).
- :func:`resolve_duplicates` — apply type-specific rules to relabel one marker
  in each duplicate pair.
- :func:`split_labelled_table` — split a labelled DataFrame into feather, body,
  and unlabelled subsets.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feather marker base names (without left_/right_ prefix)
_FEATHER_LABELS = {"wingtip", "primary", "secondary", "tailtip"}

# Body marker labels (no side prefix expected)
_BODY_LABELS = {"headpack", "backpack", "tailpack"}


def detect_duplicates(
    df: pd.DataFrame,
    *,
    frame_col: str = "frameID",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition a labelled marker table into unique, duplicate, and excess rows.

    A duplicate occurs when two rows in the same frame share the same label.
    Unlabelled rows (empty string) are always placed in the unique partition.
    This partitioning allows duplicate pairs to be resolved by
    :func:`resolve_duplicates` before analysis.

    Args:
        df: Marker table with a frame identifier column and a label column.
        frame_col: Column name for the frame identifier.
        label_col: Column name for the marker label.

    Returns:
        Tuple of three DataFrames: ``unique`` (label appears once per frame,
        plus all unlabelled rows), ``dup_pairs`` (label appears exactly twice
        per frame), and ``excess`` (label appears 3 or more times per frame).
    """
    df = df.copy()

    # Unlabelled markers are never duplicates
    labelled_mask = df[label_col] != ""
    unlabelled = df[~labelled_mask]
    labelled = df[labelled_mask]

    if labelled.empty:
        empty_df = pd.DataFrame(columns=df.columns)
        return df.copy(), empty_df, empty_df

    # Count occurrences of each (frame, label) pair
    composite = labelled[frame_col].astype(str) + "_" + labelled[label_col].astype(str)
    counts = composite.map(composite.value_counts())

    unique_labelled = labelled[counts == 1]
    dup_pairs = labelled[counts == 2]
    excess = labelled[counts >= 3]

    unique = pd.concat([unlabelled, unique_labelled], ignore_index=False)

    n_dup = len(dup_pairs) // 2 if len(dup_pairs) > 0 else 0
    n_excess = len(excess)
    logger.info(
        "  Duplicate detection: %d unique, %d duplicate pairs, %d excess rows",
        len(unique), n_dup, n_excess,
    )
    return unique, dup_pairs, excess


def resolve_duplicates(
    dup_pairs: pd.DataFrame,
    *,
    frame_col: str = "frameID",
    label_col: str = "label",
    xyz_cols: tuple[str, str, str] = ("xyz_1", "xyz_2", "xyz_3"),
    wingtip_y_threshold: float = -0.1,
) -> pd.DataFrame:
    """Resolve duplicate marker pairs by relabelling one marker in each pair.

    Applies label-specific distance heuristics to decide which marker in each
    duplicate pair should be demoted to a neighbouring label. Left/right
    side prefixes are preserved throughout. Rules applied per pair:

    - **wingtip**: the closer marker (Euclidean distance to origin) becomes
      ``primary``; if its Y is below ``wingtip_y_threshold``, it becomes
      ``secondary`` instead.
    - **primary**: the marker with smaller |Y| (closer to body midline)
      becomes ``secondary``.
    - **tailtip**: the marker further from the lateral midline (larger |X|)
      becomes ``secondary``.
    - **secondary**: the marker further from the lateral midline (larger |X|)
      becomes ``wingtip``.

    Args:
        dup_pairs: DataFrame of rows where the (frame, label) combination
            appears exactly twice, as returned by :func:`detect_duplicates`.
        frame_col: Column name for the frame identifier.
        label_col: Column name for the marker label.
        xyz_cols: Column names for the relative X, Y, Z coordinates.
        wingtip_y_threshold: Y-coordinate threshold below which a wingtip
            duplicate is demoted to secondary rather than primary. Defaults
            to -0.1.

    Returns:
        Updated copy of ``dup_pairs`` with one marker per pair relabelled.
    """
    if dup_pairs.empty:
        return dup_pairs.copy()

    result = dup_pairs.copy()
    x_col, y_col, z_col = xyz_cols

    # Group by (frame, label)
    composite = result[frame_col].astype(str) + "_" + result[label_col].astype(str)
    groups = result.groupby(composite, sort=False)

    for _key, group in groups:
        if len(group) != 2:
            continue

        idx_a, idx_b = group.index[0], group.index[1]
        current_label = result.loc[idx_a, label_col]
        base = _strip_side_prefix(current_label)

        if base == "wingtip":
            _resolve_wingtip_pair(
                result, idx_a, idx_b,
                x_col=x_col, y_col=y_col, z_col=z_col,
                label_col=label_col,
                wingtip_y_threshold=wingtip_y_threshold,
            )
        elif base == "primary":
            _resolve_primary_pair(
                result, idx_a, idx_b,
                y_col=y_col, label_col=label_col,
            )
        elif base == "tailtip":
            _resolve_tailtip_pair(
                result, idx_a, idx_b,
                x_col=x_col, label_col=label_col,
            )
        elif base == "secondary":
            _resolve_secondary_pair(
                result, idx_a, idx_b,
                x_col=x_col, label_col=label_col,
            )

    n_resolved = len(result) // 2
    logger.info("  Duplicate resolution: %d pairs resolved", n_resolved)
    return result


def split_labelled_table(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """Split a labelled marker DataFrame into feather, body, and unlabelled subsets.

    The three subsets are mutually exclusive and collectively exhaustive.
    Feather markers are wingtip, primary, secondary, and tailtip; body
    markers are headpack, backpack, and tailpack.

    Args:
        df: Marker table with a label column containing anatomical labels
            (optionally prefixed with ``left_`` or ``right_``).
        label_col: Column name containing marker labels.

    Returns:
        Dict with keys ``"feather"``, ``"body"``, and ``"unlabelled"``,
        each mapping to a DataFrame subset.
    """
    labels = df[label_col].astype(str)
    base_labels = labels.apply(_strip_side_prefix)

    feather_mask = base_labels.isin(_FEATHER_LABELS)
    body_mask = base_labels.isin(_BODY_LABELS)
    unlabelled_mask = ~feather_mask & ~body_mask

    return {
        "feather": df[feather_mask].copy(),
        "body": df[body_mask].copy(),
        "unlabelled": df[unlabelled_mask].copy(),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_side_prefix(label: str) -> str:
    """Remove ``left_`` or ``right_`` prefix from a label.

    >>> _strip_side_prefix("left_wingtip")
    'wingtip'
    >>> _strip_side_prefix("backpack")
    'backpack'
    """
    if label.startswith("left_"):
        return label[5:]
    if label.startswith("right_"):
        return label[6:]
    return label


def _replace_base_label(label: str, new_base: str) -> str:
    """Replace the base label while preserving the side prefix.

    >>> _replace_base_label("left_wingtip", "primary")
    'left_primary'
    >>> _replace_base_label("tailtip", "secondary")
    'secondary'
    """
    if label.startswith("left_"):
        return f"left_{new_base}"
    if label.startswith("right_"):
        return f"right_{new_base}"
    return new_base


def _resolve_wingtip_pair(
    df: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    label_col: str,
    wingtip_y_threshold: float,
) -> None:
    """Wingtip dups: closer → primary (or secondary if y < threshold)."""
    dist_a = np.sqrt(
        df.loc[idx_a, x_col] ** 2
        + df.loc[idx_a, y_col] ** 2
        + df.loc[idx_a, z_col] ** 2
    )
    dist_b = np.sqrt(
        df.loc[idx_b, x_col] ** 2
        + df.loc[idx_b, y_col] ** 2
        + df.loc[idx_b, z_col] ** 2
    )

    closer, _further = (idx_a, idx_b) if dist_a <= dist_b else (idx_b, idx_a)
    current_label = df.loc[closer, label_col]

    if df.loc[closer, y_col] < wingtip_y_threshold:
        df.loc[closer, label_col] = _replace_base_label(current_label, "secondary")
    else:
        df.loc[closer, label_col] = _replace_base_label(current_label, "primary")


def _resolve_primary_pair(
    df: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    *,
    y_col: str,
    label_col: str,
) -> None:
    """Primary dups: closer by y → secondary."""
    y_a = abs(df.loc[idx_a, y_col])
    y_b = abs(df.loc[idx_b, y_col])

    closer = idx_a if y_a <= y_b else idx_b
    current_label = df.loc[closer, label_col]
    df.loc[closer, label_col] = _replace_base_label(current_label, "secondary")


def _resolve_tailtip_pair(
    df: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    *,
    x_col: str,
    label_col: str,
) -> None:
    """Tailtip dups: further from midline (|x|) → secondary."""
    abs_x_a = abs(df.loc[idx_a, x_col])
    abs_x_b = abs(df.loc[idx_b, x_col])

    further = idx_a if abs_x_a >= abs_x_b else idx_b
    current_label = df.loc[further, label_col]
    df.loc[further, label_col] = _replace_base_label(current_label, "secondary")


def _resolve_secondary_pair(
    df: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    *,
    x_col: str,
    label_col: str,
) -> None:
    """Secondary dups: further from midline (|x|) → wingtip."""
    abs_x_a = abs(df.loc[idx_a, x_col])
    abs_x_b = abs(df.loc[idx_b, x_col])

    further = idx_a if abs_x_a >= abs_x_b else idx_b
    current_label = df.loc[further, label_col]
    df.loc[further, label_col] = _replace_base_label(current_label, "wingtip")
