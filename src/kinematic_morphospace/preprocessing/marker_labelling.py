"""
Body marker identification by pairwise inter-marker distances.

Labels moving markers as headpack, backpack, or tailpack based on
characteristic pairwise distances between co-located markers within each
frame. Each body pack has known inter-marker distances that serve as
fingerprints.

Reproduces the distance-bin labelling from ``run_mocap_processing.m``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distance bins (metres) — from MATLAB magic numbers
# ---------------------------------------------------------------------------

#: Headpack: 4 characteristic inter-marker distance ranges
HEADPACK_BINS: list[tuple[float, float]] = [
    (0.053, 0.056),
    (0.024, 0.025),
    (0.036, 0.037),
    (0.043, 0.044),
]

#: Backpack: 2 characteristic inter-marker distance ranges
BACKPACK_BINS: list[tuple[float, float]] = [
    (0.016, 0.017),
    (0.033, 0.034),
]

#: Tailpack: 1 characteristic inter-marker distance range
TAILPACK_BINS: list[tuple[float, float]] = [
    (0.030, 0.0325),
]


# ---------------------------------------------------------------------------
# Pairwise distance computation
# ---------------------------------------------------------------------------


def compute_pairwise_distances(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all pairwise marker distances within each frame.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with columns ``frame``, ``marker_id``, ``X``, ``Y``,
        ``Z``. Should contain only moving (non-stationary) markers.

    Returns
    -------
    pd.DataFrame
        Table with columns ``frame``, ``marker_i``, ``marker_j``,
        ``distance``.
    """
    rows = []

    for frame, group in df.groupby("frame"):
        ids = group["marker_id"].values
        xyz = group[["X", "Y", "Z"]].values

        # Skip frames with missing coordinates
        valid = ~np.isnan(xyz).any(axis=1)
        ids = ids[valid]
        xyz = xyz[valid]

        n = len(ids)
        if n < 2:
            continue

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(xyz[i] - xyz[j])
                rows.append({
                    "frame": frame,
                    "marker_i": ids[i],
                    "marker_j": ids[j],
                    "distance": dist,
                })

    result = pd.DataFrame(rows)
    logger.info("  Computed %d pairwise distances", len(result))
    return result


# ---------------------------------------------------------------------------
# Body marker labelling
# ---------------------------------------------------------------------------


def _find_markers_in_bins(
    distances: pd.DataFrame,
    bins: list[tuple[float, float]],
) -> set:
    """Find marker IDs that participate in distance pairs matching any bin."""
    matched = set()
    for d_min, d_max in bins:
        mask = (distances["distance"] >= d_min) & (distances["distance"] <= d_max)
        hits = distances[mask]
        matched.update(hits["marker_i"].values)
        matched.update(hits["marker_j"].values)
    return matched


def label_body_markers(
    df: pd.DataFrame,
    is_stationary: pd.Series | None = None,
    *,
    headpack_bins: list[tuple[float, float]] | None = None,
    backpack_bins: list[tuple[float, float]] | None = None,
    tailpack_bins: list[tuple[float, float]] | None = None,
    sample_n_frames: int | None = None,
) -> pd.Series:
    """Label markers as headpack, backpack, or tailpack by distance bins.

    For each frame (or a random sample), computes pairwise distances between
    all moving markers and checks which marker pairs fall within the
    characteristic distance bins for each body pack.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with ``frame``, ``marker_id``, ``X``, ``Y``, ``Z``.
    is_stationary : pd.Series, optional
        Boolean Series indexed by ``marker_id``; True = stationary. If
        provided, stationary markers are excluded before labelling.
    headpack_bins : list of (min, max), optional
        Override default headpack distance bins.
    backpack_bins : list of (min, max), optional
        Override default backpack distance bins.
    tailpack_bins : list of (min, max), optional
        Override default tailpack distance bins.
    sample_n_frames : int, optional
        If set, randomly sample this many frames for labelling (faster for
        long recordings). Uses majority voting across sampled frames.

    Returns
    -------
    pd.Series
        String Series indexed by ``marker_id`` with labels: ``"headpack"``,
        ``"backpack"``, ``"tailpack"``, or ``"unlabelled"``.
    """
    h_bins = headpack_bins or HEADPACK_BINS
    b_bins = backpack_bins or BACKPACK_BINS
    t_bins = tailpack_bins or TAILPACK_BINS

    # Filter to moving markers only
    if is_stationary is not None:
        moving_ids = is_stationary[~is_stationary].index
        moving = df[df["marker_id"].isin(moving_ids)].copy()
    else:
        moving = df.copy()

    # Optionally subsample frames
    if sample_n_frames is not None:
        unique_frames = moving["frame"].unique()
        if len(unique_frames) > sample_n_frames:
            rng = np.random.default_rng(42)
            sampled = rng.choice(unique_frames, size=sample_n_frames, replace=False)
            moving = moving[moving["frame"].isin(sampled)]

    # Compute pairwise distances
    distances = compute_pairwise_distances(moving)

    if distances.empty:
        all_ids = df["marker_id"].unique()
        return pd.Series("unlabelled", index=all_ids, name="body_label")

    # Find markers matching each pack's bins
    headpack_ids = _find_markers_in_bins(distances, h_bins)
    backpack_ids = _find_markers_in_bins(distances, b_bins)
    tailpack_ids = _find_markers_in_bins(distances, t_bins)

    # Resolve conflicts: a marker can only belong to one pack.
    # Priority: backpack > tailpack > headpack (backpack is most reliable).
    all_ids = df["marker_id"].unique()
    labels = pd.Series("unlabelled", index=all_ids, name="body_label")

    for mid in all_ids:
        if mid in backpack_ids:
            labels[mid] = "backpack"
        elif mid in tailpack_ids:
            labels[mid] = "tailpack"
        elif mid in headpack_ids:
            labels[mid] = "headpack"

    counts = labels.value_counts().to_dict()
    logger.info("  Body labels: %s", counts)
    return labels


# ---------------------------------------------------------------------------
# Mislabelled marker correction
# ---------------------------------------------------------------------------


def fix_mislabelled_tailpack(
    df: pd.DataFrame,
    *,
    relative_y_col: str = "xyz_2",
    label_col: str = "label",
) -> pd.DataFrame:
    """Relabel tailpack markers that are ahead of the backpack as headpack.

    In the MATLAB pipeline, tailpack markers with positive relative Y
    (i.e. ahead of the backpack) are mislabelled and should be headpack.
    Reproduces MATLAB lines 253-256 of ``run_whole_body_analysis.m``.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with relative position and label columns.
    relative_y_col : str
        Column name for relative Y position (marker - smooth backpack).
    label_col : str
        Column containing marker labels.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with corrected labels.
    """
    df = df.copy()
    is_tail = df[label_col].str.contains("tail", na=False)
    ahead = df[relative_y_col] > 0

    n_fixed = (is_tail & ahead).sum()
    df.loc[is_tail & ahead, label_col] = "headpack"

    if n_fixed > 0:
        logger.info("  Fixed %d mislabelled tailpack -> headpack", n_fixed)
    return df


# ---------------------------------------------------------------------------
# Distance-based filtering
# ---------------------------------------------------------------------------


def filter_by_distance(
    df: pd.DataFrame,
    label: str,
    min_dist: float,
    max_dist: float,
    *,
    xyz_cols: tuple[str, str, str] = ("xyz_1", "xyz_2", "xyz_3"),
    label_col: str = "label",
) -> pd.DataFrame:
    """Remove markers outside a distance range from the backpack.

    Markers of the given ``label`` whose Euclidean distance (computed from
    relative XYZ columns) falls outside ``[min_dist, max_dist]`` are
    relabelled as ``""`` (unlabelled).

    Reproduces MATLAB lines 665-707 of ``run_whole_body_analysis.m``.

    Parameters
    ----------
    df : pd.DataFrame
        Marker table with relative position columns and label column.
    label : str
        Marker label to filter (e.g. ``"backpack"``, ``"tailpack"``).
    min_dist : float
        Minimum allowed distance from backpack.
    max_dist : float
        Maximum allowed distance from backpack.
    xyz_cols : tuple of str
        Column names for relative X, Y, Z coordinates.
    label_col : str
        Column containing marker labels.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with out-of-range markers relabelled as ``""``.
    """
    df = df.copy()

    is_label = df[label_col].str.contains(label, na=False)
    if not is_label.any():
        return df

    xyz = df.loc[is_label, list(xyz_cols)].values
    dist = np.linalg.norm(xyz, axis=1)

    too_close = dist < min_dist
    too_far = dist > max_dist
    out_of_range = too_close | too_far

    if out_of_range.any():
        indices = df.index[is_label][out_of_range]
        df.loc[indices, label_col] = ""
        logger.info("  Filtered %s: removed %d (too close: %d, too far: %d)",
                     label, out_of_range.sum(), too_close.sum(), too_far.sum())

    return df
