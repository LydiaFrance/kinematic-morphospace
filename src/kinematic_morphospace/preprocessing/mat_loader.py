"""Load MATLAB .mat files and convert MATLAB table structures to pandas DataFrames.

Handles both v5/v7 (.mat via scipy) and v7.3 (.mat via mat73/h5py) formats
with automatic detection. Converts multi-column numeric fields (e.g. XYZ)
into separate _1, _2, _3 columns matching kinematic-morphospace's CSV convention.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format detection and low-level loading
# ---------------------------------------------------------------------------


def _is_hdf5(path: Path) -> bool:
    """Check whether a file is HDF5 (MATLAB v7.3) by reading its magic bytes."""
    with open(path, "rb") as f:
        return f.read(4) == b"\x89HDF"


def load_mat(path: str | Path) -> dict[str, Any]:
    """Load a MATLAB .mat file, auto-detecting v5/v7 versus v7.3 (HDF5) format.

    Args:
        path: Path to the ``.mat`` file.

    Returns:
        Dict mapping MATLAB variable names to their values (NumPy arrays,
        nested dicts, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"MAT file not found: {path}"
        raise FileNotFoundError(msg)

    if _is_hdf5(path):
        return _load_mat73(path)
    return _load_mat_scipy(path)


def _load_mat_scipy(path: Path) -> dict[str, Any]:
    """Load a v5/v7 .mat file using scipy."""
    from scipy.io import loadmat

    data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    # Remove MATLAB metadata keys
    return {k: v for k, v in data.items() if not k.startswith("__")}


def _load_mat73(path: Path) -> dict[str, Any]:
    """Load a v7.3 (HDF5) .mat file using mat73."""
    import mat73

    return mat73.loadmat(str(path))


# ---------------------------------------------------------------------------
# MATLAB table -> pandas DataFrame conversion
# ---------------------------------------------------------------------------


def _struct_to_dict(struct: Any) -> dict[str, Any]:
    """Convert a scipy struct_as_record=False object to a plain dict."""
    if hasattr(struct, "_fieldnames"):
        return {name: getattr(struct, name) for name in struct._fieldnames}
    if isinstance(struct, dict):
        return struct
    msg = f"Cannot convert {type(struct)} to dict"
    raise TypeError(msg)


def matlab_table_to_dataframe(table_struct: Any) -> pd.DataFrame:
    """Convert a MATLAB table struct to a pandas DataFrame.

    Multi-column numeric fields (e.g. an Nx3 ``XYZ`` array) are split into
    separate columns with ``_1``, ``_2``, ``_3`` suffixes, matching the
    kinematic-morphospace CSV convention. String and cell arrays are
    flattened to Python string columns.

    Args:
        table_struct: A MATLAB table loaded via scipy (as a ``mat_struct``
            object when using ``struct_as_record=False``) or via mat73 (as a
            plain dict).

    Returns:
        DataFrame with one column per MATLAB field (multi-column fields
        expanded with numeric suffixes).
    """
    data = _struct_to_dict(table_struct)
    columns: dict[str, Any] = {}

    for name, values in data.items():
        values = np.asarray(values)

        if values.dtype.kind == "O":
            # Object array — likely cell array of strings
            columns[name] = _flatten_string_array(values)
        elif values.ndim == 2 and values.shape[1] > 1:
            # Multi-column numeric field (e.g. XYZ -> XYZ_1, XYZ_2, XYZ_3)
            for col_idx in range(values.shape[1]):
                columns[f"{name}_{col_idx + 1}"] = values[:, col_idx]
        elif values.ndim == 1:
            columns[name] = values
        else:
            # Scalar or unexpected shape — store as-is
            columns[name] = values.ravel()

    return pd.DataFrame(columns)


def _flatten_string_array(arr: np.ndarray) -> list[str]:
    """Convert a numpy object array (possibly nested) to a flat list of strings."""
    result = []
    for item in arr:
        if isinstance(item, np.ndarray):
            # Nested array — take first element
            item = item.flat[0] if item.size > 0 else ""
        result.append(str(item) if item is not None else "")
    return result


# ---------------------------------------------------------------------------
# High-level loaders for the two data campaigns
# ---------------------------------------------------------------------------

# Default file names matching the MATLAB script's expected inputs
_DEFAULT_2020_FILES = {
    "backpack": "230530_meanLabelledBody.mat",
    "unlabelled": "230530_unlabelledTable.mat",
    "smooth": "230530_smoothLabelledBody.mat",
    "labelled_markers": "230530_labelledMarkersTable.mat",
}

_DEFAULT_2017_FILES = {
    "body": "220125_bodyTables.mat",
    "labelled": "220324_labelledMarkers.mat",
}


def load_2020_data(
    data_dir: str | Path,
    file_names: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all 2020 campaign .mat files and convert them to DataFrames.

    .. note::

       MATLAB ``table`` objects stored in v5/v7 .mat files are opaque MCOS
       objects that cannot be read by scipy, mat73, or any Python library.
       This function only works when the .mat files contain struct arrays
       (not MATLAB tables). For the standard hawk dataset, use
       :func:`load_intermediate_csvs` instead.

    Args:
        data_dir: Directory containing the 2020 .mat files.
        file_names: Override default file names. Keys: ``backpack``,
            ``unlabelled``, ``smooth``, ``labelled_markers``.

    Returns:
        Dict with keys ``trajectory``, ``info``, ``tail``, ``labelled``,
        ``labelled_body``, ``unlabelled``, and ``all_markers``.
    """
    data_dir = Path(data_dir)
    fnames = {**_DEFAULT_2020_FILES, **(file_names or {})}

    logger.info("Loading 2020 data from %s", data_dir)

    # --- backpack file: trajectory + info ---
    backpack_raw = load_mat(data_dir / fnames["backpack"])
    trajectory = matlab_table_to_dataframe(backpack_raw["backpackTable"])
    info = matlab_table_to_dataframe(backpack_raw["asymInfo"])
    logger.info("  backpackTable: %d rows", len(trajectory))

    # --- smooth labelled body: tail + labelled body ---
    smooth_raw = load_mat(data_dir / fnames["smooth"])
    tail = matlab_table_to_dataframe(smooth_raw["smoothTailpackTable"])
    labelled_body = matlab_table_to_dataframe(smooth_raw["asymLabelledTable"])
    logger.info("  smoothTailpackTable: %d rows", len(tail))

    # --- labelled markers ---
    labelled_raw = load_mat(data_dir / fnames["labelled_markers"])
    labelled = matlab_table_to_dataframe(labelled_raw["labelledTable"])
    logger.info("  labelledTable: %d rows", len(labelled))

    # --- unlabelled ---
    unlabelled_raw = load_mat(data_dir / fnames["unlabelled"])
    unlabelled = matlab_table_to_dataframe(unlabelled_raw["unlabelledTable"])
    all_markers = matlab_table_to_dataframe(
        unlabelled_raw["asymUnlabelledTable"]
    )
    logger.info("  unlabelledTable: %d rows", len(unlabelled))

    return {
        "trajectory": trajectory,
        "info": info,
        "tail": tail,
        "labelled": labelled,
        "labelled_body": labelled_body,
        "unlabelled": unlabelled,
        "all_markers": all_markers,
    }


def load_2017_data(
    data_dir_body: str | Path,
    data_dir_labelled: str | Path | None = None,
    file_names: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all 2017 campaign .mat files and convert them to DataFrames.

    .. note::

       MATLAB ``table`` objects stored in v5/v7 .mat files are opaque MCOS
       objects that cannot be read by scipy, mat73, or any Python library.
       This function only works when the .mat files contain struct arrays
       (not MATLAB tables). For the standard hawk dataset, use
       :func:`load_intermediate_csvs` instead.

    Args:
        data_dir_body: Directory containing the body tables .mat file
            (``220125_bodyTables.mat`` by default).
        data_dir_labelled: Directory containing the labelled markers .mat
            file (``220324_labelledMarkers.mat`` by default). Defaults to
            ``data_dir_body`` if not provided.
        file_names: Override default file names. Keys: ``body``,
            ``labelled``.

    Returns:
        Dict with keys ``trajectory``, ``smooth``, ``tail``, ``labelled``,
        and ``labelled_body``.
    """
    data_dir_body = Path(data_dir_body)
    data_dir_labelled = Path(data_dir_labelled or data_dir_body)
    fnames = {**_DEFAULT_2017_FILES, **(file_names or {})}

    logger.info("Loading 2017 data from %s", data_dir_body)

    # --- body tables ---
    body_raw = load_mat(data_dir_body / fnames["body"])
    trajectory = matlab_table_to_dataframe(body_raw["bodyTable"])
    smooth = matlab_table_to_dataframe(body_raw["smoothBodyTable"])
    tail = matlab_table_to_dataframe(body_raw["smoothTailTable"])
    logger.info("  bodyTable: %d rows", len(trajectory))

    # --- labelled markers ---
    labelled_raw = load_mat(data_dir_labelled / fnames["labelled"])
    labelled = matlab_table_to_dataframe(labelled_raw["labelledTable"])
    labelled_body = matlab_table_to_dataframe(
        labelled_raw["labelledBodyTable"]
    )
    logger.info("  labelledTable: %d rows", len(labelled))

    return {
        "trajectory": trajectory,
        "smooth": smooth,
        "tail": tail,
        "labelled": labelled,
        "labelled_body": labelled_body,
    }


# ---------------------------------------------------------------------------
# CSV-based loading (recommended path)
# ---------------------------------------------------------------------------

#: Default CSV file names matching the MATLAB-exported intermediates
_DEFAULT_CSV_FILES = {
    "traj_2017": "Traj2017.csv",
    "traj_2020": "Traj2020.csv",
    "labelled_2017": "Labelled2017.csv",
    "labelled_2020": "Labelled2020.csv",
}


def load_intermediate_csvs(
    data_dir: str | Path,
    date_prefix: str = "2024-03-24-",
    file_names: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the per-year intermediate CSVs exported by the MATLAB pipeline.

    This is the recommended entry point for the hawk dataset, because the
    original .mat files contain MATLAB ``table`` objects that are opaque to
    Python readers. MATLAB was used to export these CSVs as an intermediate
    step.

    Args:
        data_dir: Directory containing the intermediate CSV files.
        date_prefix: Prefix prepended to each default filename, e.g.
            ``"2024-03-24-"`` yields ``"2024-03-24-Traj2017.csv"``. Set
            to ``""`` to use bare filenames.
        file_names: Override default file names. Keys: ``traj_2017``,
            ``traj_2020``, ``labelled_2017``, ``labelled_2020``.

    Returns:
        Dict with keys ``traj_2017``, ``traj_2020``, ``labelled_2017``,
        and ``labelled_2020``.

    Raises:
        FileNotFoundError: If any expected CSV file is missing.
    """
    data_dir = Path(data_dir)
    names = {**_DEFAULT_CSV_FILES, **(file_names or {})}

    result: dict[str, pd.DataFrame] = {}
    for key, fname in names.items():
        path = data_dir / f"{date_prefix}{fname}"
        if not path.exists():
            msg = f"CSV file not found: {path}"
            raise FileNotFoundError(msg)
        result[key] = pd.read_csv(path)
        logger.info("  Loaded %s: %d rows from %s", key, len(result[key]), path.name)

    return result
