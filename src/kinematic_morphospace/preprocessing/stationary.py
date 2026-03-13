"""
Stationary marker detection for motion-capture data.

Identifies markers that remain stationary throughout a recording (perches,
obstacles, calibration objects) using K-means clustering on movement range,
with Calinski-Harabasz index to choose the optimal number of clusters.
Applies iterative outlier removal to refine the stationary label.

Reproduces the stationary-detection logic from ``run_mocap_processing.m``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Movement computation
# ---------------------------------------------------------------------------


def compute_marker_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-marker movement range (max - min) across all frames.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format marker table with columns ``marker_id``, ``X``, ``Y``,
        ``Z``.

    Returns
    -------
    pd.DataFrame
        One row per marker with columns ``marker_id``, ``range_X``,
        ``range_Y``, ``range_Z``, ``total_range``.
    """
    grouped = df.groupby("marker_id")[["X", "Y", "Z"]]
    ranges = grouped.max() - grouped.min()
    ranges.columns = ["range_X", "range_Y", "range_Z"]
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
    """Label markers as stationary or moving using K-means + Calinski-Harabasz.

    The algorithm:

    1. Compute total movement range per marker.
    2. Try K-means with k=2..5, pick k with highest Calinski-Harabasz score.
    3. Assign the cluster with lowest mean range as "stationary".
    4. Within the stationary cluster, iteratively remove outliers (markers
       whose range exceeds mean + ``outlier_std_factor`` * std).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format marker table with ``marker_id``, ``X``, ``Y``, ``Z``.
    threshold : float
        Fallback threshold for stationary if clustering fails (total range
        must be below this value).
    n_outlier_passes : int
        Number of outlier-removal iterations within the stationary cluster.
    outlier_std_factor : float
        Multiplier on standard deviation for outlier detection.

    Returns
    -------
    pd.Series
        Boolean Series indexed by ``marker_id``; True = stationary.
    """
    movement = compute_marker_movement(df)
    ranges = movement["total_range"].values.reshape(-1, 1)
    marker_ids = movement["marker_id"].values

    if len(marker_ids) < 5:
        logger.warning("  Too few markers (%d) for clustering, using threshold", len(marker_ids))
        is_stationary = pd.Series(
            movement["total_range"].values < threshold,
            index=marker_ids,
            name="stationary",
        )
        return is_stationary

    # Try k=2..5, pick best Calinski-Harabasz
    best_score = -1
    best_labels = None

    for k in range(2, min(6, len(marker_ids))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(ranges)
        if len(set(labels)) < 2:
            continue
        score = calinski_harabasz_score(ranges, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    if best_labels is None:
        logger.warning("  Clustering failed, using threshold fallback")
        is_stationary = pd.Series(
            movement["total_range"].values < threshold,
            index=marker_ids,
            name="stationary",
        )
        return is_stationary

    # Identify stationary cluster (lowest mean range)
    cluster_means = {}
    for c in set(best_labels):
        cluster_means[c] = ranges[best_labels == c].mean()
    stationary_cluster = min(cluster_means, key=cluster_means.get)

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
    logger.info("  Detected %d stationary / %d moving markers", n_stat, len(result) - n_stat)
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
    """Assign semantic labels to stationary markers by Y-coordinate position.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format marker table with ``marker_id`` and ``Y``.
    is_stationary : pd.Series
        Boolean Series indexed by ``marker_id`` (from
        :func:`detect_stationary_markers`).
    y_ranges : dict, optional
        Mapping of label → (y_min, y_max). Defaults to
        :data:`DEFAULT_OBJECT_RANGES`.

    Returns
    -------
    pd.Series
        String Series indexed by ``marker_id`` with labels:
        ``"left_perch"``, ``"right_perch"``, ``"obstacle"``, ``"moving"``,
        or ``"stationary_unknown"``.
    """
    if y_ranges is None:
        y_ranges = DEFAULT_OBJECT_RANGES

    # Compute median Y per marker
    median_y = df.groupby("marker_id")["Y"].median()

    labels = pd.Series("moving", index=is_stationary.index, name="object_label")

    for marker_id in is_stationary.index:
        if not is_stationary[marker_id]:
            continue

        y = median_y.get(marker_id, np.nan)
        if np.isnan(y):
            labels[marker_id] = "stationary_unknown"
            continue

        assigned = False
        for label, (y_min, y_max) in y_ranges.items():
            if y_min <= y <= y_max:
                labels[marker_id] = label
                assigned = True
                break

        if not assigned:
            labels[marker_id] = "stationary_unknown"

    logger.info("  Object labels: %s", labels.value_counts().to_dict())
    return labels
