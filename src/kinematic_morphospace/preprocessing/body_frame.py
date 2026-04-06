"""Body pitch estimation from backpack marker positions.

Uses PCA on per-frame marker deviations from the centroid to find the
body's principal axis, then computes the pitch angle as the angle between
this axis and the vertical (Z) direction.

Reproduces the body-pitch logic from ``run_mocap_processing.m``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_body_pitch(
    df: pd.DataFrame,
    body_labels: pd.Series | None = None,
    *,
    min_markers: int = 3,
) -> pd.DataFrame:
    """Estimate body pitch angle per frame from backpack marker positions.

    For each frame, selects backpack markers, computes deviations from
    the centroid, and eigendecomposes the covariance matrix. The principal
    eigenvector (largest eigenvalue) gives the body's main axis; pitch is
    the angle between that axis and the vertical [0, 0, 1], in degrees.

    Args:
        df: Marker table with columns ``frame``, ``marker_id``, ``X``,
            ``Y``, ``Z``.
        body_labels: Series indexed by ``marker_id`` containing anatomical
            labels. Only markers labelled ``"backpack"`` are used. If None,
            all markers in the table are used.
        min_markers: Minimum number of backpack markers required per frame.
            Frames with fewer valid markers receive ``body_pitch = NaN``.
            Defaults to 3 (matching the MATLAB implementation).

    Returns:
        DataFrame with columns ``frame``, ``body_pitch`` (degrees),
        ``normal_X``, ``normal_Y``, ``normal_Z``.
    """
    # Filter to backpack markers
    if body_labels is not None:
        bp_ids = body_labels[body_labels == "backpack"].index
        bp = df[df["marker_id"].isin(bp_ids)].copy()
    else:
        bp = df.copy()

    results = []

    for frame, group in bp.groupby("frame"):
        xyz = group[["X", "Y", "Z"]].dropna().values

        if len(xyz) < min_markers:
            results.append({
                "frame": frame,
                "body_pitch": np.nan,
                "normal_X": np.nan,
                "normal_Y": np.nan,
                "normal_Z": np.nan,
            })
            continue

        # Centroid and deviation vectors
        centroid = xyz.mean(axis=0)
        deviations = xyz - centroid

        # Covariance matrix and eigendecomposition
        cov_matrix = np.cov(deviations.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Eigenvector with largest eigenvalue (last, since eigh sorts ascending)
        normal_vector = eigenvectors[:, -1]

        # Ensure consistent orientation (positive Z component)
        if normal_vector[2] < 0:
            normal_vector = -normal_vector

        # Pitch angle: angle between normal vector and vertical [0,0,1]
        cos_angle = np.clip(np.dot(np.array([0, 0, 1]), normal_vector), -1.0, 1.0)
        pitch_deg = np.degrees(np.arccos(cos_angle))

        results.append({
            "frame": frame,
            "body_pitch": pitch_deg,
            "normal_X": normal_vector[0],
            "normal_Y": normal_vector[1],
            "normal_Z": normal_vector[2],
        })

    result_df = pd.DataFrame(results)

    n_valid = result_df["body_pitch"].notna().sum()
    logger.info(
        "  Body pitch: %d/%d frames computed (%.1f%% valid)",
        n_valid, len(result_df),
        100 * n_valid / max(1, len(result_df)),
    )
    return result_df
