"""Polygon-based marker labelling using boundary definitions.

Labels unlabelled motion-capture markers by testing whether their relative
positions (in both XY and YZ planes) fall within predefined polygon boundaries
for each bird and marker type. Uses ``matplotlib.path.Path.contains_points()``
as the vectorized equivalent of MATLAB's ``inpolygon``.

Reproduces steps 5-6 of ``run_whole_body_analysis.m``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_polygon_boundaries(
    mat_path: str | Path,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Load per-bird, per-marker polygon boundary definitions from a .mat file.

    Reads a nested MATLAB struct of the form
    ``areaDefs.<bird>.<marker_type>.flightPhase(1).XY`` and
    ``areaDefs.<bird>.<marker_type>.flightPhase(1).YZ`` to extract 2D
    polygon vertices used for labelling unlabelled markers.

    Args:
        mat_path: Path to the ``.mat`` file containing polygon boundary
            definitions (e.g. ``230530_AreaDefs.mat``).

    Returns:
        Nested dict with structure
        ``boundaries[bird][marker_type][plane]`` → (N, 2) array of polygon
        vertices, where ``plane`` is ``"XY"`` or ``"YZ"``.

    Raises:
        KeyError: If the ``areaDefs`` variable is not found in the .mat file.
    """
    from .mat_loader import load_mat

    data = load_mat(mat_path)

    if "areaDefs" not in data:
        msg = f"Expected 'areaDefs' variable in {mat_path}"
        raise KeyError(msg)

    area_defs = data["areaDefs"]
    boundaries: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for bird_name, bird_data in _iter_struct(area_defs):
        boundaries[bird_name] = {}
        for marker_type, marker_data in _iter_struct(bird_data):
            flight_phase = _get_flight_phase(marker_data)
            xy = _extract_array(flight_phase, "XY")
            yz = _extract_array(flight_phase, "YZ")
            if xy is not None and yz is not None:
                boundaries[bird_name][marker_type] = {"XY": xy, "YZ": yz}

    n_polys = sum(
        len(markers) for markers in boundaries.values()
    )
    logger.info("  Loaded %d polygon boundaries for %d birds",
                n_polys, len(boundaries))
    return boundaries


def _iter_struct(obj: Any) -> list[tuple[str, Any]]:
    """Iterate over MATLAB struct fields (works for dict or object)."""
    if isinstance(obj, dict):
        return [(k, v) for k, v in obj.items() if not k.startswith("_")]
    # scipy.io loadmat returns numpy void arrays for structs
    if hasattr(obj, "dtype") and obj.dtype.names:
        return [(name, obj[name]) for name in obj.dtype.names]
    return []


def _get_flight_phase(marker_data: Any) -> Any:
    """Extract flightPhase(1) from nested MATLAB struct."""
    if isinstance(marker_data, dict):
        fp = marker_data.get("flightPhase", marker_data)
        if isinstance(fp, (list, np.ndarray)) and len(fp) > 0:
            return fp[0]
        return fp
    # Handle scipy structured arrays
    if hasattr(marker_data, "dtype") and "flightPhase" in (marker_data.dtype.names or []):
        fp = marker_data["flightPhase"]
        if hasattr(fp, "flat"):
            fp = fp.flat[0]
        return fp
    return marker_data


def _extract_array(obj: Any, field: str) -> np.ndarray | None:
    """Extract a named (N, 2) array from a struct-like object."""
    if isinstance(obj, dict):
        arr = obj.get(field)
    elif hasattr(obj, "dtype") and obj.dtype.names and field in obj.dtype.names:
        arr = obj[field]
        if hasattr(arr, "flat"):
            arr = arr.flat[0]
    else:
        return None

    if arr is not None:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
    return None


def label_by_polygons(
    df: pd.DataFrame,
    boundaries: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    bird_col: str = "seqID",
    bird_id_map: dict[str, str] | None = None,
    xyz_cols: tuple[str, str, str] = ("xyz_1", "xyz_2", "xyz_3"),
    label_col: str = "label",
    lateralise: bool = False,
) -> pd.DataFrame:
    """Label unlabelled markers by testing them against polygon boundaries.

    For each bird, tests all currently unlabelled markers against the
    per-marker-type polygon boundaries in both the XY and YZ planes. A
    marker receives a label only when it falls inside both polygons, using
    ``matplotlib.path.Path.contains_points()`` as a vectorised equivalent
    of MATLAB's ``inpolygon``.

    Args:
        df: Marker table with relative position columns and a label column.
            Unlabelled markers must have ``label_col`` set to ``""``.
        boundaries: Polygon boundaries as returned by
            :func:`load_polygon_boundaries`.
        bird_col: Column used to identify which bird each row belongs to,
            matched against boundary dict keys.
        bird_id_map: Maps bird identifiers from ``bird_col`` to boundary
            dict keys, e.g. ``{"01": "Drogon", "03": "Ruby"}``.
        xyz_cols: Column names for the relative X, Y, Z coordinates.
        label_col: Column name containing marker labels.
        lateralise: If True, prefix assigned labels with ``"left_"``
            (X < 0) or ``"right_"`` (X >= 0). Defaults to False.

    Returns:
        Updated copy of ``df`` with newly labelled markers.
    """
    from matplotlib.path import Path as MplPath

    df = df.copy()
    x_col, y_col, z_col = xyz_cols

    total_labelled = 0

    for bird_key, marker_boundaries in boundaries.items():
        # Find rows belonging to this bird
        if bird_id_map:
            bird_mask = df[bird_col].apply(
                lambda s, bk=bird_key: bird_id_map.get(
                    _extract_bird_prefix(s), ""
                ) == bk
            )
        else:
            bird_mask = df[bird_col].str.contains(bird_key, na=False)

        for marker_type, planes in marker_boundaries.items():
            xy_verts = planes["XY"]
            yz_verts = planes["YZ"]

            xy_path = MplPath(xy_verts)
            yz_path = MplPath(yz_verts)

            # Test all unlabelled markers for this bird
            unlabelled = bird_mask & (df[label_col] == "")
            if not unlabelled.any():
                continue

            xy_points = df.loc[unlabelled, [x_col, y_col]].values
            yz_points = df.loc[unlabelled, [y_col, z_col]].values

            in_xy = xy_path.contains_points(xy_points)
            in_yz = yz_path.contains_points(yz_points)

            inside_both = in_xy & in_yz
            label_indices = df.index[unlabelled][inside_both]

            if lateralise:
                x_vals = df.loc[label_indices, x_col]
                left_idx = x_vals[x_vals < 0].index
                right_idx = x_vals[x_vals >= 0].index
                df.loc[left_idx, label_col] = f"left_{marker_type}"
                df.loc[right_idx, label_col] = f"right_{marker_type}"
            else:
                df.loc[label_indices, label_col] = marker_type

            n = inside_both.sum()
            total_labelled += n
            if n > 0:
                logger.debug("  %s/%s: labelled %d markers",
                             bird_key, marker_type, n)

    logger.info("  Polygon labelling: %d markers labelled", total_labelled)
    return df


def _extract_bird_prefix(seq_id: str) -> str:
    """Extract bird prefix (e.g. '01') from a sequence ID like '01_09_001'."""
    return seq_id[:2] if len(seq_id) >= 2 else seq_id
