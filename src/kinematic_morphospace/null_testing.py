"""Utility functions to support PCA null-model analyses with sequence-aware resampling.

These helpers ensure that mirrored left/right frames remain paired
throughout statistical procedures.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def flatten_frames(frames: np.ndarray) -> np.ndarray:
    """Flatten a 3-D marker array to 2-D, or return a copy if already 2-D.

    Args:
        frames: Array of shape ``(n_frames, n_markers, n_dims)`` or
            ``(n_frames, n_features)``.

    Returns:
        Copy with shape ``(n_frames, n_features)``.

    Raises:
        ValueError: If the input array does not have 2 or 3 dimensions.
    """
    array = np.asarray(frames)
    if array.ndim == 3:
        n_frames, n_markers, n_dims = array.shape
        return array.reshape(n_frames, n_markers * n_dims)
    if array.ndim == 2:
        return array.copy()
    raise ValueError(f"Unexpected frame array shape {array.shape}")


def ensure_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator, passing through an existing one unchanged.

    Args:
        seed: Integer seed for a new Generator, an existing Generator to
            pass through, or None for an unseeded Generator.

    Returns:
        NumPy Generator instance.
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
    """Sanity-check that the frame array lines up with the frame metadata.

    Args:
        frames: Array of shape ``(n_frames, ...)``.
        frame_info: DataFrame with one row per frame.
        expected_columns: Column names that must be present in ``frame_info``.
            Defaults to ``("seqID", "BirdID", "Obstacle", "Left")``.

    Raises:
        ValueError: If the lengths mismatch or required columns are missing.
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
    """Derive grouping vectors used to keep mirrored frames paired during resampling.

    Args:
        frame_info: Per-frame metadata DataFrame.
        sequence_column: Column identifying flight sequences. Defaults to
            ``"seqID"``.
        side_column: Column indicating left (1) or right (0) side. Defaults
            to ``"Left"``.
        individual_column: Column identifying individual birds. Defaults to
            ``"BirdID"``.
        condition_column: Column indicating whether an obstacle was present.
            Defaults to ``"Obstacle"``.

    Returns:
        Dict containing arrays: ``seq_ids``, ``unique_seq_ids``,
        ``seq_index``, ``side_flag``, ``bird_ids``, and ``obstacles``.
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
    """Build a list-of-indices lookup for quick sequence masking.

    Args:
        seq_index: Integer array of shape ``(n_frames,)`` mapping each
            frame to a unique sequence index.

    Returns:
        List of arrays, one per unique sequence, each containing the
        frame indices belonging to that sequence.
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
    """Resample sequence indices with replacement and expand to frame indices.

    Sequences (rather than individual frames) are resampled so that
    mirrored left/right frames stay together, preserving the bilateral
    pairing structure throughout the bootstrap.

    Args:
        seq_index: Integer array mapping each frame to a sequence index.
        size: Number of sequences to resample. Defaults to the number of
            unique sequences (standard bootstrap).
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Boolean mask of shape ``(n_frames,)`` indicating selected frames.
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
    """Permute labels at the sequence level to break group-level associations.

    Shuffles labels across sequences rather than individual frames, so
    within-sequence label consistency is preserved under the null.

    Args:
        labels: Per-frame label array of shape ``(n_frames,)``.
        seq_index: Integer array mapping each frame to a sequence index.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Permuted label array of shape ``(n_frames,)`` with sequence-level
        shuffling applied.
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
    """Summarise an array using mean, standard deviation, and percentile statistics.

    Args:
        values: 1-D array of numeric values to summarise.
        decimals: Number of decimal places to round to. Defaults to 4.
        percentiles: Percentile values to include in the summary. Defaults
            to ``(2.5, 50.0, 97.5)``.

    Returns:
        Series with keys ``mean``, ``std``, and one key per percentile
        formatted as ``p{value}`` (e.g. ``p2.5``).
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
    """Produce a tidy summary table for cumulative explained variance curves.

    Args:
        cev: Array of cumulative explained variance values. May be 1-D
            ``(n_components,)`` for a single run or 2-D
            ``(n_runs, n_components)`` for multiple runs.
        components: Component indices (1-based) to include. Defaults to all.
        decimals: Number of decimal places to round to. Defaults to 4.

    Returns:
        DataFrame with one row per component and columns ``mean``, ``std``,
        and percentile columns.
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
    """Compute pairwise distances between markers for each frame, vectorised.

    Args:
        frames: Array of shape ``(n_frames, n_markers, 3)`` containing 3-D
            marker coordinates.
        max_markers: If set, only use the first ``max_markers`` markers.
            Useful for tests or when comparing subsets. Defaults to None.
        sort_per_frame: If True, sort the distances within each frame so the
            representation is label-free (order-invariant). Defaults to False.
        descending: If True and ``sort_per_frame`` is True, sort in descending
            order. Defaults to True.

    Returns:
        Array of shape ``(n_frames, n_pairs)`` where
        ``n_pairs = n_markers * (n_markers - 1) // 2``. If
        ``sort_per_frame`` is True, distances within each row are sorted.

    Raises:
        ValueError: If ``frames`` is not 3-D or ``max_markers`` < 2.
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

    diffs = []
    for i in range(n_markers - 1):
        d = array[:, i, None, :] - array[:, i + 1:, :]
        diffs.append(d)
    diffs = np.concatenate(diffs, axis=1)

    distances = np.linalg.norm(diffs, axis=2)

    if sort_per_frame:
        distances = np.sort(distances, axis=1)
        if descending:
            distances = distances[:, ::-1]

    return distances


def load_missing_marker_dataset(
    npz_path: Path,
) -> Tuple[np.ndarray, pd.DataFrame, Sequence[str]]:
    """Load the labelled dataset with missing markers from an npz file.

    Args:
        npz_path: Path to the ``labelled_markers_with_missing.npz`` file.

    Returns:
        Tuple of (markers, info, columns) where markers is the marker array,
        info is a per-frame metadata DataFrame, and columns is the sequence
        of marker column names.
    """
    loaded = np.load(npz_path, allow_pickle=True)
    markers = loaded["marker_data"]
    info = pd.DataFrame(loaded["info_data"], columns=loaded["info_column_names"])
    columns = loaded["marker_column_names"]
    return markers, info, columns


def prepare_missing_marker_dataset(
    npz_path: Path | str,
    wingspan_path: Path | str,
    *,
    straight_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the missing-marker dataset, scale by wingspan, and convert to unilateral.

    Args:
        npz_path: Path to ``labelled_markers_with_missing.npz``.
        wingspan_path: Path to ``TotalWingspans.yml``.
        straight_only: If True (default), keep only non-obstacle frames.

    Returns:
        Tuple of (missing_unilateral, missing_bilateral, missing_info) where
        missing_unilateral has shape ``(2*N, 4, 3)`` (left then right,
        x-mirrored), missing_bilateral has shape ``(M, 8, 3)`` (the full
        scaled bilateral array), and missing_info is a per-frame metadata
        DataFrame.
    """
    import yaml
    from .pca_reconstruct import to_unilateral

    bilateral, info, _cols = load_missing_marker_dataset(Path(npz_path))

    with open(wingspan_path) as f:
        wingspans = yaml.safe_load(f)

    hawk_names = {1: "Drogon", 2: "Rhaegal", 3: "Ruby", 4: "Toothless", 5: "Charmander"}
    bird_ids = info["BirdID"].astype(int).values
    years = info["Year"].astype(int).values

    for hawk_id, hawk_name in hawk_names.items():
        for year in [2017, 2020]:
            if year not in wingspans.get(hawk_name, {}):
                continue
            mask = (bird_ids == hawk_id) & (years == year)
            if mask.any():
                bilateral[mask] /= wingspans[hawk_name][year]

    if straight_only:
        keep = (info["Obstacle"].astype(int) == 0).values
        bilateral_subset = bilateral[keep]
    else:
        bilateral_subset = bilateral

    unilateral = to_unilateral(bilateral_subset)

    return unilateral, bilateral, info


def load_unlabelled_csv(
    csv_path: Path | str,
) -> Tuple[Dict[int, np.ndarray], List[int], List[int]]:
    """Load the unlabelled marker CSV and group by frame.

    Args:
        csv_path: Path to ``unlabelled_markers.csv``.

    Returns:
        Tuple of (frame_groups, labelled_ids, unlabelled_ids) where
        frame_groups maps frameID → ``(n_markers, 3)`` coordinate array,
        labelled_ids contains frame IDs with exactly 8 markers, and
        unlabelled_ids contains frame IDs with fewer than 8 markers.
    """
    df = pd.read_csv(csv_path)
    frame_counts = df.groupby("frameID").size()
    labelled_ids = frame_counts[frame_counts == 8].index.tolist()
    unlabelled_ids = frame_counts[frame_counts != 8].index.tolist()

    xyz_cols = [c for c in df.columns if c.startswith("rot_xyz")]
    frame_groups: Dict[int, np.ndarray] = {
        fid: group[xyz_cols].to_numpy().reshape(-1, 3)
        for fid, group in df.groupby("frameID")
    }
    return frame_groups, labelled_ids, unlabelled_ids


def principal_cosines(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    *,
    modes: int,
    return_angles: bool = False
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Compute principal cosines between the leading modes of two bases.

    Principal cosines (the singular values of ``Qa.T @ Qb``) measure the
    alignment between two PCA subspaces.  A value of 1 means the subspaces
    are identical; 0 means they are orthogonal.  This is the standard
    metric for comparing PCA stability across bootstrap resamples or data
    subsets.

    Args:
        basis_a: Array of shape ``(n_features, n_components_a)`` whose
            columns are basis vectors (e.g. ``pca.components_.T``).
        basis_b: Array of shape ``(n_features, n_components_b)`` whose
            columns are basis vectors.
        modes: Number of leading modes to compare (k).
        return_angles: If True, also return ``arccos(cosines)`` in radians.
            Defaults to False.

    Returns:
        Array of length ``modes`` with principal cosines in descending order.
        Padded with ``np.nan`` where ``modes`` exceeds the available
        components. If ``return_angles`` is True, returns a tuple of
        (cosines, angles_in_radians).

    Raises:
        ValueError: If the inputs are not 2-D or do not share the same
            number of feature rows.
    """
    if basis_a.ndim != 2 or basis_b.ndim != 2:
        raise ValueError("basis_a and basis_b must be 2-D arrays")

    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError("Bases must have the same number of feature rows (same ambient dimension)")

    A = np.asarray(basis_a, dtype=np.float64)
    B = np.asarray(basis_b, dtype=np.float64)

    k_avail = min(A.shape[1], B.shape[1], modes)
    if k_avail == 0:
        cosines = np.full(modes, np.nan, dtype=np.float64)
        if return_angles:
            return cosines, np.full_like(cosines, np.nan)
        return cosines

    Qa, _ = np.linalg.qr(A[:, :k_avail], mode="reduced")
    Qb, _ = np.linalg.qr(B[:, :k_avail], mode="reduced")

    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)

    s = np.clip(s, -1.0, 1.0)

    out = np.full(modes, np.nan, dtype=np.float64)
    out[:k_avail] = s

    if return_angles:
        angles = np.arccos(out)
        return out, angles
    return out


def random_relabel_frames(
    frames: np.ndarray,
    *,
    swap_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly permute marker labels within a fraction of frames.

    Used for robustness checks: shuffling marker identities within frames
    tests whether the PCA structure is driven by consistent labelling or
    genuine shape variation.

    Args:
        frames: Marker array of shape ``(n_frames, n_markers, 3)``.
        swap_fraction: Fraction of frames in which markers are permuted
            (e.g. 0.1 = 10 % of frames).
        rng: NumPy Generator for reproducibility.

    Returns:
        Copy of ``frames`` with marker order randomly permuted within the
        selected subset of frames.
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
    """Relabel markers across frames using a simple velocity-prediction heuristic.

    At each frame, predicts marker positions from the previous frame plus a
    smoothed velocity estimate, then uses the Hungarian algorithm to match
    predicted to observed positions. Assignments with cost above
    ``max_displacement`` are blocked to prevent large-displacement swaps.

    Args:
        frames: Marker array of shape ``(n_frames, n_markers, 3)``.
        max_displacement: Maximum allowable distance between a predicted and
            observed position for assignment. Assignments above this threshold
            are given a prohibitive cost. Defaults to 0.4.

    Returns:
        Relabelled marker array of the same shape as ``frames``.
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
