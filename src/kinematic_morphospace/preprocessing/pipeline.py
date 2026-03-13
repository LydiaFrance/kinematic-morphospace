"""
Full preprocessing pipeline that orchestrates MAT loading, harmonisation,
calibration, and shape-table construction.

Reproduces the MATLAB script ``fix_data_2024_03_23.m`` as a single
``run_preprocessing()`` call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import numpy as np

from .calibration import apply_time_offsets, calibrate_position, calibrate_time
from .harmonise import harmonise_labelled, harmonise_trajectory
from .mat_loader import load_2017_data, load_2020_data, load_intermediate_csvs
from .shape_tables import create_bilateral_table, create_unilateral_table

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for the full preprocessing pipeline.

    All paths and physical constants are configurable, with defaults
    matching the original MATLAB script.
    """

    # --- Data directories ---
    data_dir_2020: str | Path = ""
    data_dir_2017_body: str | Path = ""
    data_dir_2017_labelled: str | Path | None = None
    output_dir: str | Path | None = None

    # --- File name overrides ---
    file_names_2020: dict[str, str] = field(default_factory=dict)
    file_names_2017: dict[str, str] = field(default_factory=dict)

    # --- Physical constants ---
    perch_height: float = 1.25
    jump_dist: float = 8.3
    tolerances: tuple[float, ...] = (0.02, 0.05, 0.2)

    # --- Processing options ---
    include_unrotated: bool = False
    """Also produce unilateral/bilateral tables with ``xyz`` coordinates."""


def run_preprocessing(
    config: PreprocessingConfig,
) -> dict[str, pd.DataFrame]:
    """Run the full preprocessing pipeline from raw .mat files.

    Loads raw .mat files from both campaigns, harmonises column names and
    metadata, calibrates position and time (2020 only), combines years,
    and constructs unilateral and bilateral shape tables.

    .. note::

       This requires .mat files with struct-based variables (not MATLAB
       ``table`` objects). For the standard hawk dataset, use
       :func:`run_from_csvs` instead, which loads from the per-year
       intermediate CSVs that MATLAB already exported.

    Parameters
    ----------
    config : PreprocessingConfig
        Pipeline configuration.

    Returns
    -------
    dict[str, pd.DataFrame]
        Output tables keyed by name:

        - ``"trajectory"`` — combined trajectory table
        - ``"labelled"`` — combined labelled marker table (long format)
        - ``"unilateral"`` — unilateral shape table (wide, rotated)
        - ``"bilateral"`` — bilateral shape table (wide, rotated)

        If ``config.include_unrotated`` is True, also includes:

        - ``"unilateral_unrotated"`` — unilateral with ``xyz`` coordinates
        - ``"bilateral_unrotated"`` — bilateral with ``xyz`` coordinates
    """
    logger.info("=" * 60)
    logger.info("kinematic-morphospace Preprocessing Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading raw .mat files")

    data_2020 = load_2020_data(
        config.data_dir_2020,
        file_names=config.file_names_2020 or None,
    )
    data_2017 = load_2017_data(
        config.data_dir_2017_body,
        config.data_dir_2017_labelled,
        file_names=config.file_names_2017 or None,
    )

    # ------------------------------------------------------------------
    # 2. Harmonise trajectory tables
    # ------------------------------------------------------------------
    logger.info("Step 2: Harmonising trajectory tables")

    traj_2020 = harmonise_trajectory(
        data_2020["trajectory"],
        year=2020,
        info_df=data_2020["info"],
        tail_df=data_2020["tail"],
    )
    traj_2017 = harmonise_trajectory(
        data_2017["trajectory"],
        year=2017,
        tail_df=data_2017["tail"],
        smooth_df=data_2017["smooth"],
    )

    # ------------------------------------------------------------------
    # 3. Calibrate 2020 trajectory (position + time)
    # ------------------------------------------------------------------
    logger.info("Step 3: Calibrating 2020 trajectory")

    traj_2020 = calibrate_position(
        traj_2020, perch_height=config.perch_height
    )
    traj_2020, time_offsets = calibrate_time(
        traj_2020,
        jump_dist=config.jump_dist,
        tolerances=config.tolerances,
    )

    # ------------------------------------------------------------------
    # 4. Combine trajectory tables
    # ------------------------------------------------------------------
    logger.info("Step 4: Combining trajectory tables")

    trajectory = pd.concat(
        [traj_2017, traj_2020], ignore_index=True
    )
    logger.info("  Combined trajectory: %d rows", len(trajectory))

    # ------------------------------------------------------------------
    # 5. Harmonise labelled tables
    # ------------------------------------------------------------------
    logger.info("Step 5: Harmonising labelled tables")

    labelled_2020 = harmonise_labelled(
        data_2020["labelled"],
        year=2020,
        info_df=data_2020["info"],
    )
    labelled_2017 = harmonise_labelled(
        data_2017["labelled"],
        year=2017,
    )

    # ------------------------------------------------------------------
    # 6. Calibrate 2020 labelled (position + time)
    # ------------------------------------------------------------------
    logger.info("Step 6: Calibrating 2020 labelled")

    labelled_2020 = calibrate_position(
        labelled_2020, perch_height=config.perch_height
    )
    labelled_2020 = apply_time_offsets(labelled_2020, time_offsets)

    # ------------------------------------------------------------------
    # 7. Combine labelled tables
    # ------------------------------------------------------------------
    logger.info("Step 7: Combining labelled tables")

    labelled = pd.concat(
        [labelled_2017, labelled_2020], ignore_index=True
    )
    logger.info("  Combined labelled: %d rows", len(labelled))

    # ------------------------------------------------------------------
    # 8. Create shape tables
    # ------------------------------------------------------------------
    logger.info("Step 8: Creating shape tables")

    unilateral = create_unilateral_table(labelled, coord_prefix="rot_xyz")
    bilateral = create_bilateral_table(labelled, coord_prefix="rot_xyz")

    results = {
        "trajectory": trajectory,
        "labelled": labelled,
        "unilateral": unilateral,
        "bilateral": bilateral,
    }

    if config.include_unrotated:
        logger.info("Step 8b: Creating unrotated shape tables")
        results["unilateral_unrotated"] = create_unilateral_table(
            labelled, coord_prefix="xyz"
        )
        results["bilateral_unrotated"] = create_bilateral_table(
            labelled, coord_prefix="xyz"
        )

    # ------------------------------------------------------------------
    # 9. Save CSVs (if output_dir specified)
    # ------------------------------------------------------------------
    if config.output_dir:
        save_csvs(results, config.output_dir)

    logger.info("=" * 60)
    logger.info("Preprocessing complete")
    for name, df in results.items():
        logger.info("  %s: %d rows x %d cols", name, len(df), len(df.columns))
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CSV-based pipeline (recommended)
# ---------------------------------------------------------------------------


def run_from_csvs(
    csv_dir: str | Path,
    output_dir: str | Path | None = None,
    include_unrotated: bool = False,
    date_prefix: str = "2024-03-24-",
) -> dict[str, pd.DataFrame]:
    """Run shape-table construction from MATLAB-exported intermediate CSVs.

    This is the recommended entry point for the hawk dataset. It loads the
    per-year trajectory and labelled CSVs, concatenates them, drops NaN
    coordinate rows, adds ``VertDistance``, and builds unilateral + bilateral
    shape tables.

    Parameters
    ----------
    csv_dir : str or Path
        Directory containing the intermediate CSVs (e.g.
        ``2024-03-24-Traj2017.csv``).
    output_dir : str or Path, optional
        If provided, save output CSVs here.
    include_unrotated : bool
        Also produce unilateral/bilateral tables with ``xyz`` coordinates.
    date_prefix : str
        Prefix on the CSV file names (default ``"2024-03-24-"``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Output tables keyed by name:

        - ``"trajectory"`` — combined trajectory table
        - ``"labelled"`` — combined labelled marker table
        - ``"unilateral"`` — unilateral shape table (wide, rotated)
        - ``"bilateral"`` — bilateral shape table (wide, rotated)

        If *include_unrotated* is True, also includes:

        - ``"unilateral_unrotated"`` — unilateral with ``xyz`` coordinates
        - ``"bilateral_unrotated"`` — bilateral with ``xyz`` coordinates
    """
    logger.info("=" * 60)
    logger.info("kinematic-morphospace CSV Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load intermediate CSVs
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading intermediate CSVs from %s", csv_dir)
    csvs = load_intermediate_csvs(csv_dir, date_prefix=date_prefix)

    # ------------------------------------------------------------------
    # 2. Concatenate trajectory tables
    # ------------------------------------------------------------------
    logger.info("Step 2: Concatenating trajectory tables")
    trajectory = pd.concat(
        [csvs["traj_2017"], csvs["traj_2020"]], ignore_index=True
    )
    logger.info("  Combined trajectory: %d rows", len(trajectory))

    # ------------------------------------------------------------------
    # 3. Concatenate labelled tables and clean
    # ------------------------------------------------------------------
    logger.info("Step 3: Concatenating labelled tables")
    labelled = pd.concat(
        [csvs["labelled_2017"], csvs["labelled_2020"]], ignore_index=True
    )
    logger.info("  Combined labelled (raw): %d rows", len(labelled))

    # Drop rows with any NaN (matching MATLAB — primarily removes rows
    # with NaN time values from incomplete marker observations)
    n_before = len(labelled)
    labelled = labelled.dropna()
    logger.info("  Dropped %d rows with NaN (%d remain)", n_before - len(labelled), len(labelled))

    # Add VertDistance from smooth backpack Z if not already present
    if "VertDistance" not in labelled.columns:
        if "backpack_smooth_XYZ_3" in labelled.columns:
            labelled["VertDistance"] = labelled["backpack_smooth_XYZ_3"]
        else:
            labelled["VertDistance"] = np.nan

    # ------------------------------------------------------------------
    # 4. Build shape tables
    # ------------------------------------------------------------------
    logger.info("Step 4: Creating shape tables")
    unilateral = create_unilateral_table(labelled, coord_prefix="rot_xyz")
    bilateral = create_bilateral_table(labelled, coord_prefix="rot_xyz")

    results: dict[str, pd.DataFrame] = {
        "trajectory": trajectory,
        "labelled": labelled,
        "unilateral": unilateral,
        "bilateral": bilateral,
    }

    if include_unrotated:
        logger.info("Step 4b: Creating unrotated shape tables")
        results["unilateral_unrotated"] = create_unilateral_table(
            labelled, coord_prefix="xyz"
        )
        results["bilateral_unrotated"] = create_bilateral_table(
            labelled, coord_prefix="xyz"
        )

    # ------------------------------------------------------------------
    # 5. Save CSVs (if output_dir specified)
    # ------------------------------------------------------------------
    if output_dir:
        save_csvs(results, output_dir)

    logger.info("=" * 60)
    logger.info("CSV pipeline complete")
    for name, df in results.items():
        logger.info("  %s: %d rows x %d cols", name, len(df), len(df.columns))
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

#: Default output file names matching the MATLAB-produced CSVs
_DEFAULT_OUTPUT_NAMES = {
    "trajectory": "FullTraj.csv",
    "labelled": "FullLabelled.csv",
    "unilateral": "FullUnilateralMarkers.csv",
    "bilateral": "FullBilateralMarkers.csv",
    "unilateral_unrotated": "FullUnilateralMarkersNoRot.csv",
    "bilateral_unrotated": "FullBilateralMarkersNoRot.csv",
}


def save_csvs(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    file_names: dict[str, str] | None = None,
    date_prefix: str | None = None,
) -> dict[str, Path]:
    """Write output DataFrames to CSV files.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Tables to write, keyed by name (matching :data:`_DEFAULT_OUTPUT_NAMES`).
    output_dir : str or Path
        Directory to write CSVs into.
    file_names : dict[str, str], optional
        Override default file names.
    date_prefix : str, optional
        If provided, prepend this to each file name (e.g. ``"2024-03-24-"``).

    Returns
    -------
    dict[str, Path]
        Mapping of table name to the written file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {**_DEFAULT_OUTPUT_NAMES, **(file_names or {})}
    prefix = date_prefix or ""

    written: dict[str, Path] = {}

    for key, df in tables.items():
        fname = names.get(key, f"{key}.csv")
        path = output_dir / f"{prefix}{fname}"
        df.to_csv(path, index=False)
        written[key] = path
        logger.info("  Saved %s -> %s (%d rows)", key, path.name, len(df))

    return written
