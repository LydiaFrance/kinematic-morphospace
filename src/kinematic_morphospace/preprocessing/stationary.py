"""Stationary marker detection for motion-capture data.

Identifies markers that remain stationary throughout a recording (perches,
obstacles, calibration objects) using K-means clustering on movement range,
with Calinski-Harabasz index to choose the optimal number of clusters.
Applies iterative outlier removal to refine the stationary label.

Reproduces the stationary-detection logic from ``run_mocap_processing.m``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Movement computation
# ---------------------------------------------------------------------------


def compute_marker_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the coordinate-wise movement range for each marker across all frames.

    The range (max - min) for each of X, Y, Z is summed into ``total_range``,
    which is used as the primary feature for stationary marker detection.

    Args:
        df: Long-format marker table with columns ``marker_id``, ``X``,
            ``Y``, ``Z``.

    Returns:
        DataFrame with one row per marker and columns ``marker_id``,
        ``range_X``, ``range_Y``, ``range_Z``, and ``total_range``.
    """
    grouped = df.groupby("marker_id")[["X", "Y", "Z"]]
    max_vals = grouped.max().to_numpy()
    min_vals = grouped.min().to_numpy()
    ranges = pd.DataFrame(
        max_vals - min_vals,
        index=grouped.max().index,
        columns=pd.Index(["range_X", "range_Y", "range_Z"]),
    )
    ranges["total_range"] = ranges[["range_X", "range_Y", "range_Z"]].sum(axis=1)
    return ranges.reset_index()


# ---------------------------------------------------------------------------
# Stationary detection
# ---------------------------------------------------------------------------


def detect_stationary_markers(
    df: pd.DataFrame,
    threshold: float = 0.001,
    n_outlier_passes: int = 3,
    outlier_std_factor: float = 2.0,
) -> pd.Series:
    """Identify stationary markers (perches, obstacles) using K-means clustering.

    Clusters markers by their total movement range using K-means (k=2-5),
    selecting the number of clusters with the highest Calinski-Harabasz
    score. The cluster with the lowest mean range is labelled stationary.
    Outlier markers within the stationary cluster are iteratively removed
    based on a standard-deviation threshold.

    Falls back to a simple threshold comparison if fewer than 5 markers are
    present or if clustering fails.

    Args:
        df: Long-format marker table with columns ``marker_id``, ``X``,
            ``Y``, ``Z``.
        threshold: Fallback movement threshold in metres. Markers with total
            range below this value are considered stationary when clustering
            cannot be performed. Defaults to 0.001.
        n_outlier_passes: Number of iterative outlier-removal passes within
            the stationary cluster. Defaults to 3.
        outlier_std_factor: Standard-deviation multiplier used to define
            outlier markers within the stationary cluster. Defaults to 2.0.

    Returns:
        Boolean Series indexed by ``marker_id``; True means stationary.
    """
    movement = compute_marker_movement(df)
    ranges = movement["total_range"].to_numpy().reshape(-1, 1)
    marker_ids = movement["marker_id"].values

    if len(marker_ids) < 5:
        logger.warning(
            "  Too few markers (%d) for clustering, using threshold",
            len(marker_ids),
        )
        return pd.Series(
            movement["total_range"].to_numpy() < threshold,
            index=marker_ids,
            name="stationary",
        )

    # Try k=2..5, pick best Calinski-Harabasz
    best_score = -1
    best_labels = None

    for k in range(2, min(6, len(marker_ids))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)  # type: ignore[arg-type]
        labels = km.fit_predict(ranges)
        if len(set(labels)) < 2:
            continue
        score = calinski_harabasz_score(ranges, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    if best_labels is None:
        logger.warning("  Clustering failed, using threshold fallback")
        return pd.Series(
            movement["total_range"].to_numpy() < threshold,
            index=marker_ids,
            name="stationary",
        )

    # Identify stationary cluster (lowest mean range)
    cluster_means: dict[int, Any] = {}
    for c in set(best_labels):
        cluster_means[c] = ranges[best_labels == c].mean()
    stationary_cluster = min(cluster_means, key=lambda c: cluster_means[c])

    is_stationary = best_labels == stationary_cluster

    # Iterative outlier removal within stationary cluster
    for _pass in range(n_outlier_passes):
        stat_ranges = ranges[is_stationary].flatten()
        if len(stat_ranges) < 2:
            break
        cutoff = stat_ranges.mean() + outlier_std_factor * stat_ranges.std()
        # Remove markers above cutoff from stationary set
        new_stationary = is_stationary & (ranges.flatten() <= cutoff)
        if np.array_equal(new_stationary, is_stationary):
            break
        is_stationary = new_stationary

    result = pd.Series(is_stationary, index=marker_ids, name="stationary")
    n_stat = result.sum()
    logger.info(
        "  Detected %d stationary / %d moving markers",
        n_stat,
        len(result) - n_stat,
    )
    return result


# ---------------------------------------------------------------------------
# Fixed-object labelling
# ---------------------------------------------------------------------------

#: Default Y-coordinate ranges (in metres) for identifying fixed objects
#: in the 2020 hawk flight arena.
DEFAULT_OBJECT_RANGES: dict[str, tuple[float, float]] = {
    "left_perch": (-7.5, -5.5),
    "right_perch": (1.5, 3.5),
    "obstacle": (-3.0, -1.0),
}


def label_fixed_objects(
    df: pd.DataFrame,
    is_stationary: pd.Series,
    y_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.Series:
    """Assign semantic labels to stationary markers based on their Y-axis position.

    Stationary markers are mapped to named arena objects (perches, obstacle)
    by comparing their median Y position against known Y-coordinate ranges.
    Moving markers receive the label ``"moving"``; stationary markers that
    fall outside all defined ranges receive ``"stationary_unknown"``.

    Args:
        df: Long-format marker table with columns ``marker_id`` and ``Y``.
        is_stationary: Boolean Series indexed by ``marker_id``, as returned
            by :func:`detect_stationary_markers`.
        y_ranges: Mapping of object label to (y_min, y_max) Y-coordinate
            range in metres. Defaults to :data:`DEFAULT_OBJECT_RANGES`.

    Returns:
        String Series indexed by ``marker_id`` with values
        ``"left_perch"``, ``"right_perch"``, ``"obstacle"``, ``"moving"``,
        or ``"stationary_unknown"``.
    """
    if y_ranges is None:
        y_ranges = DEFAULT_OBJECT_RANGES

    # Compute median Y per marker
    median_y = df.groupby("marker_id")["Y"].median()

    labels = pd.Series("moving", index=is_stationary.index, name="object_label")

    for marker_id in is_stationary.index:
        if not is_stationary.loc[marker_id]:
            continue

        y = median_y.get(marker_id, np.nan)
        if y is not None and np.isnan(y):  # type: ignore[arg-type]
            labels[marker_id] = "stationary_unknown"
            continue

        assigned = False
        for label, (y_min, y_max) in y_ranges.items():
            if y is not None and y_min <= y <= y_max:
                labels[marker_id] = label
                assigned = True
                break

        if not assigned:
            labels[marker_id] = "stationary_unknown"

    logger.info("  Object labels: %s", labels.value_counts().to_dict())
    return labels
