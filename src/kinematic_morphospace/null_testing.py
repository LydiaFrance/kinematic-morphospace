"""
Utility functions to support PCA null-model analyses with sequence-aware
resampling. These helpers ensure that mirrored left/right frames remain paired
throughout statistical procedures.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def flatten_frames(frames: np.ndarray) -> np.ndarray:
    """
    Flatten marker frames to two dimensions.

    Parameters
    ----------
    frames : np.ndarray
        Either (n_frames, n_markers, n_dims) or (n_frames, n_features).

    Returns
    -------
    np.ndarray
        A copy with shape (n_frames, n_features).
    """
    array = np.asarray(frames)
    if array.ndim == 3:
        n_frames, n_markers, n_dims = array.shape
        return array.reshape(n_frames, n_markers * n_dims)
    if array.ndim == 2:
        return array.copy()
    raise ValueError(f"Unexpected frame array shape {array.shape}")


def ensure_rng(seed: int | None = None) -> np.random.Generator:
    """
    Provide a NumPy Generator using a fixed seed when supplied.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def validate_frame_alignment(
    frames: np.ndarray,
    frame_info: pd.DataFrame,
    *,
    expected_columns: Sequence[str] = ("seqID", "BirdID", "Obstacle", "Left"),
) -> None:
    """
    Sanity-check that the frame array lines up with the frame metadata.

    Raises
    ------
    ValueError
        If the lengths mismatch or required columns are missing.
    """
    if frames.shape[0] != len(frame_info):
        raise ValueError(
            "Frame count mismatch: "
            f"{frames.shape[0]} frames versus {len(frame_info)} metadata rows."
        )

    missing = [column for column in expected_columns if column not in frame_info.columns]
    if missing:
        raise ValueError(
            "Frame metadata missing required columns: "
            + ", ".join(sorted(missing))
        )


def prepare_sequence_groups(
    frame_info: pd.DataFrame,
    *,
    sequence_column: str = "seqID",
    side_column: str = "Left",
    individual_column: str = "BirdID",
    condition_column: str = "Obstacle",
) -> Dict[str, np.ndarray]:
    """
    Derive grouping vectors used to keep mirrored frames paired.
    """
    seq_ids = frame_info[sequence_column].astype(str).to_numpy()
    unique_seq_ids, seq_inverse = np.unique(seq_ids, return_inverse=True)

    side_flag = frame_info[side_column].to_numpy()
    bird_ids = frame_info[individual_column].astype(str).to_numpy()
    obstacles = frame_info[condition_column].to_numpy()

    return {
        "seq_ids": seq_ids,
        "unique_seq_ids": unique_seq_ids,
        "seq_index": seq_inverse,
        "side_flag": side_flag.astype(int, copy=False),
        "bird_ids": bird_ids,
        "obstacles": obstacles,
    }


def sequence_lookup(seq_index: np.ndarray) -> List[np.ndarray]:
    """
    Build a list-of-indices lookup for quick sequence masking.
    """
    buckets: Dict[int, List[int]] = defaultdict(list)
    for frame_idx, seq_idx in enumerate(seq_index):
        buckets[int(seq_idx)].append(frame_idx)
    return [np.asarray(indices, dtype=int) for indices in buckets.values()]


def grouped_bootstrap_indices(
    seq_index: np.ndarray,
    *,
    size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Resample sequence indices with replacement and expand to frame indices.
    """
    generator = ensure_rng(seed)
    unique_indices = np.unique(seq_index)
    if size is None:
        size = unique_indices.size
    sampled_seq = generator.choice(unique_indices, size=size, replace=True)
    mask = np.zeros(seq_index.shape[0], dtype=bool)
    for seq_id in sampled_seq:
        mask |= seq_index == seq_id
    return mask


def grouped_permutation_labels(
    labels: np.ndarray,
    seq_index: np.ndarray,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """
    Permute labels at the sequence level.
    """
    generator = ensure_rng(seed)
    unique_seq = np.unique(seq_index)
    seq_labels = np.zeros(unique_seq.shape[0], dtype=labels.dtype)
    for idx, seq_id in enumerate(unique_seq):
        frame_positions = np.flatnonzero(seq_index == seq_id)
        if frame_positions.size == 0:
            raise ValueError(f"Sequence id {seq_id} has no associated frames.")
        seq_labels[idx] = labels[frame_positions[0]]
    permuted = generator.permutation(seq_labels)
    return permuted[seq_index]


def summarise_distribution(
    values: np.ndarray,
    *,
    decimals: int = 4,
    percentiles: Sequence[float] = (2.5, 50.0, 97.5),
) -> pd.Series:
    """
    Summarise an array using mean and percentile statistics.
    """
    array = np.asarray(values)
    summary: Dict[str, float] = {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
    }
    for perc in percentiles:
        summary[f"p{perc:g}"] = float(np.percentile(array, perc))
    series = pd.Series(summary)
    return series.round(decimals)


def summarise_cumulative_variance(
    cev: np.ndarray,
    *,
    components: Sequence[int] | None = None,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Produce a tidy summary table for cumulative explained variance curves.
    """
    array = np.asarray(cev)
    if array.ndim == 1:
        array = array[None, :]
    n_components = array.shape[1]
    if components is None:
        components = list(range(1, n_components + 1))
    matrix = []
    for comp in components:
        idx = comp - 1
        column = summarise_distribution(array[:, idx], decimals=decimals)
        column.name = f"k={comp}"
        matrix.append(column)
    return pd.concat(matrix, axis=1).T


def pairwise_distance_features(
    frames: np.ndarray,
    *,
    max_markers: int | None = None,
    sort_per_frame: bool = False,
    descending: bool = True
) -> np.ndarray:
    """
    Compute pairwise distances between markers for each frame, vectorised.

    Args:
        frames: (n_frames, n_markers, 3) array of 3D marker coords.
        max_markers: if set, only use the first max_markers markers (useful for tests).
        sort_per_frame: if True, sort the distances within each frame (makes
                        representation label-free).
        descending: if True and sort_per_frame True, sort in descending order.
        
    Returns:
        distances: (n_frames, n_pairs) array where n_pairs = n_markers*(n_markers-1)//2.
                   If sort_per_frame=True, the distances in each row are sorted.
    """
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("Expected frames with shape (n_frames, n_markers, 3)")
    n_frames, n_markers, _ = array.shape

    if max_markers is not None:
        if max_markers < 2:
            raise ValueError("max_markers must be >= 2")
        array = array[:, :max_markers, :]
        n_markers = max_markers

    # number of unique unordered pairs
    pair_count = n_markers * (n_markers - 1) // 2
    # compute pairwise differences efficiently:
    # create all pair offsets using broadcasting
    # result shape will be (n_frames, pair_count, 3)
    diffs = []
    for i in range(n_markers - 1):
        # differences between marker i and markers i+1..end
        d = array[:, i, None, :] - array[:, i + 1 :, :]
        diffs.append(d)  # shape (n_frames, n_markers-i-1, 3)
    diffs = np.concatenate(diffs, axis=1)  # (n_frames, pair_count, 3)

    distances = np.linalg.norm(diffs, axis=2)  # (n_frames, pair_count)

    # Optional per-frame sorting to remove pair identity
    if sort_per_frame:
        distances = np.sort(distances, axis=1)
        if descending:
            distances = distances[:, ::-1]

    return distances

def load_missing_marker_dataset(
    npz_path: Path,
) -> Tuple[np.ndarray, pd.DataFrame, Sequence[str]]:
    """
    Load the labelled dataset with missing markers.
    """
    loaded = np.load(npz_path, allow_pickle=True)
    markers = loaded["marker_data"]
    info = pd.DataFrame(loaded["info_data"], columns=loaded["info_column_names"])
    columns = loaded["marker_column_names"]
    return markers, info, columns


def principal_cosines(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    *,
    modes: int,
    return_angles: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal cosines between the leading modes of two bases.

    Args:
        basis_a: (n_features, n_components_a) array (columns = basis vectors).
        basis_b: (n_features, n_components_b) array (columns = basis vectors).
        modes: number of leading modes to compare (k).
        return_angles: if True return (cosines, angles_in_radians).

    Returns:
        cosines: 1D array length `modes` with principal cosines in descending order.
                 If modes > min(n_components_a, n_components_b) the tail is filled with np.nan.
        (optional) angles: arccos(cosines) (nan where cosines is nan).
    """
    if basis_a.ndim != 2 or basis_b.ndim != 2:
        raise ValueError("basis_a and basis_b must be 2-D arrays")

    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError("Bases must have the same number of feature rows (same ambient dimension)")

    # ensure float64 for stable numeric ops
    A = np.asarray(basis_a, dtype=np.float64)
    B = np.asarray(basis_b, dtype=np.float64)

    # effective k we can compute
    k_avail = min(A.shape[1], B.shape[1], modes)
    if k_avail == 0:
        # nothing to compute; return nans
        cosines = np.full(modes, np.nan, dtype=np.float64)
        if return_angles:
            return cosines, np.full_like(cosines, np.nan)
        return cosines

    # reduced/ economy QR to orthonormalise the selected columns
    Qa, _ = np.linalg.qr(A[:, :k_avail], mode="reduced")
    Qb, _ = np.linalg.qr(B[:, :k_avail], mode="reduced")

    # singular values of Qa.T @ Qb are the principal cosines (descending)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)

    # numeric safety: clip to [-1,1]
    s = np.clip(s, -1.0, 1.0)

    # build output array length `modes`, pad with nan if requested modes > available
    out = np.full(modes, np.nan, dtype=np.float64)
    out[:k_avail] = s

    if return_angles:
        angles = np.arccos(out)  # nan where out is nan
        return out, angles
    return out

def random_relabel_frames(
    frames: np.ndarray,
    *,
    swap_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly relabel markers within frames for robustness checks.
    """
    n_frames, n_markers, _ = frames.shape
    shuffled = frames.copy()
    n_swap = int(np.round(swap_fraction * n_frames))
    indices = rng.choice(n_frames, size=n_swap, replace=False)
    for frame_idx in indices:
        shuffled[frame_idx] = shuffled[frame_idx, rng.permutation(n_markers), :]
    return shuffled


def relabel_with_predictor(
    frames: np.ndarray,
    *,
    max_displacement: float = 0.4,
) -> np.ndarray:
    """
    Simple predictor-based relabelling heuristic.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    n_frames, n_markers, _ = frames.shape
    relabelled = np.zeros_like(frames)
    relabelled[0] = frames[0].copy()
    previous = relabelled[0].copy()
    velocity = np.zeros_like(previous)

    for frame_idx in range(1, n_frames):
        predicted = previous + velocity
        current = frames[frame_idx]
        costs = cdist(predicted, current)
        costs[costs > max_displacement] = 1e9
        rows, cols = linear_sum_assignment(costs)
        assigned = np.zeros_like(current)
        for row, col in zip(rows, cols):
            assigned[row] = current[col]
        relabelled[frame_idx] = assigned
        new_velocity = relabelled[frame_idx] - previous
        velocity = 0.6 * velocity + 0.4 * new_velocity
        previous = relabelled[frame_idx].copy()
    return relabelled




