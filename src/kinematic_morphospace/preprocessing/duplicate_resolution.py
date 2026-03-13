"""
Duplicate marker detection and resolution for wing labelling.

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
    """Partition rows by duplicate status within each frame.

    A "duplicate" means two rows share the same ``frame_col`` and
    ``label_col`` value. Unlabelled rows (empty string) are always placed
    in *unique*.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with frame and label columns.
    frame_col, label_col : str
        Column names for frame identifier and marker label.

    Returns
    -------
    unique : pd.DataFrame
        Rows whose label appears exactly once in their frame (plus unlabelled).
    dup_pairs : pd.DataFrame
        Rows whose label appears exactly twice in their frame.
    excess : pd.DataFrame
        Rows whose label appears 3+ times in their frame.
    """
    df = df.copy()

    # Unlabelled markers are never duplicates
    labelled_mask = df[label_col] != ""
    unlabelled = df[~labelled_mask]
    labelled = df[labelled_mask]

    if labelled.empty:
        return df.copy(), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

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
    """Resolve duplicate pairs by relabelling one marker in each pair.

    Rules (applied per pair, preserving left_/right_ prefix):

    - **wingtip** dups: closer marker (by Euclidean distance to origin)
      becomes *primary*; if its y < ``wingtip_y_threshold``, becomes
      *secondary* instead.
    - **primary** dups: closer marker (by y-coordinate) becomes *secondary*.
    - **tailtip** dups: marker further from midline (larger ``|x|``) becomes
      *secondary*.
    - **secondary** dups: marker further from midline (larger ``|x|``) becomes
      *wingtip*.

    Parameters
    ----------
    dup_pairs : pd.DataFrame
        Rows with exactly 2 occurrences of their (frame, label) combination.
    frame_col, label_col : str
        Column names.
    xyz_cols : tuple of str
        Column names for X, Y, Z coordinates.
    wingtip_y_threshold : float
        Y-coordinate threshold for wingtip→secondary demotion.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with duplicates resolved.
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
    """Split a labelled DataFrame into feather, body, and unlabelled subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with a label column.
    label_col : str
        Column containing marker labels.

    Returns
    -------
    dict
        Keys ``"feather"``, ``"body"``, ``"unlabelled"``, each mapping to a
        DataFrame subset. The three subsets are mutually exclusive and
        collectively exhaustive.
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
