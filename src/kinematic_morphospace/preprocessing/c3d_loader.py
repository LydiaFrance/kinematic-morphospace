"""
C3D file parsing and file-list construction.

Loads raw C3D motion-capture files using ``ezc3d`` and extracts marker
trajectories as DataFrames. Also provides utilities for building a file list
from a directory of C3D files and filtering for recordings with wing markers.

Requires the ``[preprocessing]`` extra (``ezc3d>=1.5``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C3D loading
# ---------------------------------------------------------------------------


def load_c3d(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Load a single C3D file and return marker trajectories + metadata.

    Parameters
    ----------
    path : str or Path
        Path to a ``.c3d`` file.

    Returns
    -------
    df : pd.DataFrame
        Long-format table with columns:
        ``frame``, ``marker_id``, ``marker_label``, ``X``, ``Y``, ``Z``,
        ``residual``.
    metadata : dict
        Recording metadata including ``frame_rate``, ``n_frames``,
        ``n_markers``, ``marker_labels``, ``first_frame``, ``last_frame``.

    Raises
    ------
    FileNotFoundError
        If the C3D file does not exist.
    ImportError
        If ``ezc3d`` is not installed.
    """
    path = Path(path)
    if not path.exists():
        msg = f"C3D file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        import ezc3d
    except ImportError as exc:
        msg = (
            "ezc3d is required for C3D loading. "
            "Install with: pip install 'kinematic-morphospace[preprocessing]'"
        )
        raise ImportError(msg) from exc

    c3d = ezc3d.c3d(str(path))

    # Extract metadata
    header = c3d["header"]
    params = c3d["parameters"]

    frame_rate = header["points"]["frame_rate"]
    first_frame = header["points"]["first_frame"]
    last_frame = header["points"]["last_frame"]
    n_frames = last_frame - first_frame + 1

    marker_labels = params["POINT"]["LABELS"]["value"]
    n_markers = len(marker_labels)

    metadata = {
        "frame_rate": frame_rate,
        "n_frames": n_frames,
        "n_markers": n_markers,
        "marker_labels": marker_labels,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "source_file": str(path),
    }

    # Extract point data: shape (4, n_markers, n_frames)
    # Row 0=X, 1=Y, 2=Z, 3=residual
    points = c3d["data"]["points"]

    rows = []
    for marker_idx in range(n_markers):
        label = marker_labels[marker_idx]
        for frame_idx in range(n_frames):
            frame_num = first_frame + frame_idx
            rows.append({
                "frame": frame_num,
                "marker_id": marker_idx,
                "marker_label": label,
                "X": points[0, marker_idx, frame_idx],
                "Y": points[1, marker_idx, frame_idx],
                "Z": points[2, marker_idx, frame_idx],
                "residual": points[3, marker_idx, frame_idx],
            })

    df = pd.DataFrame(rows)

    # Replace zero-residual entries (missing markers) with NaN
    missing = df["residual"] < 0
    df.loc[missing, ["X", "Y", "Z"]] = np.nan

    logger.info(
        "  Loaded %s: %d markers x %d frames @ %.0f Hz",
        path.name, n_markers, n_frames, frame_rate,
    )

    return df, metadata


# ---------------------------------------------------------------------------
# File list construction
# ---------------------------------------------------------------------------

#: Regex for parsing 2020 hawk C3D filenames.
#: Format: ``{date}_{bird}_{distance}m_{imu}[_nobackpack][_Obstacle]_Trial{nn}.c3d``
_FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{6})_(?P<bird>Charmander|Drogon|Ruby|Toothless)"
    r"_(?P<distance>\d+)m"
    r"_(?P<imu>IMUweighton|IMUweightoff|noIMU)"
    r"(?P<nobackpack>_nobackpack)?"
    r"(?P<obstacle>_Obstacle)?"
    r"_Trial(?P<trial>\d+)"
    r"\.c3d$",
    re.IGNORECASE,
)

#: Bird name → numeric ID mapping (from MATLAB pipeline).
BIRD_ID_MAP: dict[str, int] = {
    "Drogon": 1,
    "Rhaegal": 2,
    "Ruby": 3,
    "Toothless": 4,
    "Charmander": 5,
}


def build_file_list(mocap_folder: str | Path) -> pd.DataFrame:
    """Scan a directory recursively for C3D files and extract metadata from filenames.

    Parameters
    ----------
    mocap_folder : str or Path
        Root directory containing ``.c3d`` files (scans subdirectories).

    Returns
    -------
    pd.DataFrame
        Table with columns: ``path``, ``filename``, ``date``, ``bird``,
        ``bird_id``, ``distance``, ``imu``, ``obstacle``, ``nobackpack``,
        ``trial``.
    """
    mocap_folder = Path(mocap_folder)
    rows = []

    for p in sorted(mocap_folder.rglob("*.c3d")):
        m = _FILENAME_PATTERN.match(p.name)
        if m:
            bird_name = m.group("bird")
            imu_raw = m.group("imu")
            rows.append({
                "path": str(p),
                "filename": p.name,
                "date": m.group("date"),
                "bird": bird_name,
                "bird_id": BIRD_ID_MAP[bird_name],
                "distance": int(m.group("distance")),
                "imu": imu_raw.lower() != "noimu",
                "obstacle": m.group("obstacle") is not None,
                "nobackpack": m.group("nobackpack") is not None,
                "trial": int(m.group("trial")),
            })
        else:
            logger.warning("  Skipping unrecognised filename: %s", p.name)

    df = pd.DataFrame(rows)
    logger.info("  Found %d C3D files in %s", len(df), mocap_folder)
    return df


def filter_file_list(file_list: pd.DataFrame) -> pd.DataFrame:
    """Keep only recordings with a backpack (exclude ``nobackpack`` sessions).

    All 2020 wing-marker sessions have wing markers by default; the MATLAB
    pipeline's "wings" flag corresponded to the ``1130`` date sessions.
    The only sessions to exclude are those explicitly tagged ``nobackpack``.

    Parameters
    ----------
    file_list : pd.DataFrame
        Output of :func:`build_file_list`.

    Returns
    -------
    pd.DataFrame
        Filtered copy retaining only rows with ``nobackpack=False``.
    """
    mask = ~file_list["nobackpack"]
    filtered = file_list[mask].copy()
    logger.info(
        "  Filtered: %d → %d recordings (backpack only)",
        len(file_list), len(filtered),
    )
    return filtered
