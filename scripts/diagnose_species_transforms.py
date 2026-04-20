"""Diagnostic script for cross-species transformation analysis.

Systematically processes every species in the Harvey et al. dataset through
the full cross-species transformation pipeline and reports:

- Marker positions and norms after the full preprocessing chain
- NaN-poisoned markers (which silently corrupt positions before transformation)
- Near-zero markers (norm < 0.01 m) that trigger ValueError in
  compute_transformation_matrix()
- Per-marker scale factors and rotation angles
- Pass/Fail status with a human-readable issue description

Usage::

    uv run python scripts/diagnose_species_transforms.py

Output is printed to the console and saved to
``scripts/species_transform_diagnostics.csv``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from morphing_birds import Animal3D

# ---------------------------------------------------------------------------
# Resolve project root so the script works from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

WING_FILE = PROJECT_ROOT / "data" / "harvey_2022" / "specimen_winginfo.csv"
BODY_FILE = PROJECT_ROOT / "data" / "harvey_2022" / "specimen_bodyinfo.csv"
HAWK_SHAPE = PROJECT_ROOT / "data" / "mean_hawk_shape.csv"
OUTPUT_CSV = PROJECT_ROOT / "scripts" / "species_transform_diagnostics.csv"

# Threshold below which a marker norm is considered "near-zero"
NEAR_ZERO_THRESHOLD = 0.01  # metres
# Thresholds outside which a scale factor is considered "extreme"
SCALE_LOW = 0.1
SCALE_HIGH = 10.0
# Rotation threshold above which a marker transformation is flagged as "high rotation"
ROTATION_WARN_DEG = 45.0
# Scale variance threshold: flag if max/min scale ratio exceeds this
SCALE_VARIANCE_WARN = 4.0

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from kinematic_morphospace import (
    load_harvey_data,
    select_max_wingspan_row,
    clean_body_data,
    process_body_bird_id,
    merge_bird_data,
    filter_marker_columns,
    set_new_origin_and_axes,
    compute_derived_markers,
    check_and_fix_shoulder_distance,
    fix_leftright_sign,
)
from kinematic_morphospace.species_transform import compute_transformation_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Marker names as used in the bilateral marker columns
MARKER_NAMES = [
    "wingtip",
    "primary",
    "secondary",
    "tailtip",
]


def _right_markers_from_df(df: pd.DataFrame, row_idx: int) -> np.ndarray:
    """Extract right-side derived marker positions as an (n_markers, 3) array.

    Returns the same array that ``Animal3D.right_markers`` would expose after
    calling ``integrate_dataframe_to_bird3D``, i.e. the exact values passed to
    ``compute_transformation_matrix`` during transformation.

    Only the four *moving* markers are returned (wingtip, primary, secondary,
    tailtip) because these are the ones that pass through the per-marker
    transformation; fixed markers (shoulder, tailbase, hood) are handled
    separately and are not scaled/rotated by the block-diagonal matrix.

    Args:
        df: Fully preprocessed merged DataFrame.
        row_idx: Row index to extract.

    Returns:
        Array of shape ``(n_markers, 3)``.
    """
    coords = []
    for name in MARKER_NAMES:
        x = df.at[row_idx, f"right_{name}_x"]
        y = df.at[row_idx, f"right_{name}_y"]
        z = df.at[row_idx, f"right_{name}_z"]
        coords.append([x, y, z])
    return np.array(coords, dtype=float)


def _nan_mask(markers: np.ndarray) -> np.ndarray:
    """Return a boolean array indicating which markers contain any NaN."""
    return np.any(np.isnan(markers), axis=1)


def _near_zero_mask(markers: np.ndarray, threshold: float = NEAR_ZERO_THRESHOLD) -> np.ndarray:
    """Return a boolean array indicating which markers have near-zero norm."""
    norms = np.linalg.norm(markers, axis=1)
    return norms < threshold


def _rotation_angle_rad(source: np.ndarray, target: np.ndarray) -> float:
    """Compute the rotation angle (radians) between two 3-D vectors.

    Mirrors the angle computation inside ``compute_transformation_matrix``
    so that the diagnostic reports the same value that would be used in the
    actual transformation.

    Args:
        source: Source vector (3,).
        target: Target vector (3,).

    Returns:
        Angle in radians, or NaN if either vector has zero length.
    """
    ns = np.linalg.norm(source)
    nt = np.linalg.norm(target)
    if ns == 0 or nt == 0:
        return float("nan")
    dot = np.dot(source, target)
    cos_angle = np.clip(dot / (ns * nt), -1.0, 1.0)
    return float(np.arccos(cos_angle))


# ---------------------------------------------------------------------------
# Per-species analysis
# ---------------------------------------------------------------------------

def analyse_species(
    species_name: str,
    species_rows: pd.DataFrame,
    hawk_markers: np.ndarray,
) -> dict:
    """Run the full diagnostic for a single species.

    Picks the first fully-populated row for the species (or the first row if
    none have complete data), extracts derived marker positions, and attempts
    the transformation.

    Args:
        species_name: Common name of the target species.
        species_rows: Subset of the merged DataFrame belonging to this species.
        hawk_markers: Reference hawk right-side markers, shape ``(n_markers, 3)``.

    Returns:
        Dictionary with all diagnostic fields for one CSV row.
    """
    # Use the first row (already sorted by wingspan descending via select_max_wingspan_row)
    row_idx: int = species_rows.index[0]  # type: ignore[assignment]

    wingspan_cm = species_rows.at[row_idx, "wing_span_cm"]

    # ------------------------------------------------------------------
    # Extract derived marker positions for the target species
    # ------------------------------------------------------------------
    target_markers = _right_markers_from_df(species_rows, row_idx)

    nan_flags = _nan_mask(target_markers)
    n_nan_markers = int(nan_flags.sum())
    nan_marker_names = [MARKER_NAMES[i] for i in range(len(MARKER_NAMES)) if nan_flags[i]]

    norms = np.where(nan_flags, np.nan, np.linalg.norm(target_markers, axis=1))
    near_zero_flags = ~nan_flags & (norms < NEAR_ZERO_THRESHOLD)
    n_near_zero = int(near_zero_flags.sum())
    near_zero_names = [MARKER_NAMES[i] for i in range(len(MARKER_NAMES)) if near_zero_flags[i]]

    # ------------------------------------------------------------------
    # Attempt transformation, collect per-marker statistics
    # ------------------------------------------------------------------
    scale_factors: list[float] = []
    rotation_angles_deg: list[float] = []
    transformation_ok = True
    error_message = ""
    extreme_scale_markers: list[str] = []

    for i, name in enumerate(MARKER_NAMES):
        src = hawk_markers[i]
        tgt = target_markers[i]

        # NaN in target propagates silently — flag it
        if nan_flags[i]:
            scale_factors.append(float("nan"))
            rotation_angles_deg.append(float("nan"))
            continue

        try:
            T = compute_transformation_matrix(src, tgt)
        except ValueError as exc:
            transformation_ok = False
            error_message = f"Marker '{name}': {exc}"
            scale_factors.append(float("nan"))
            rotation_angles_deg.append(float("nan"))
            continue

        # Extract scale: the determinant of T / rotation_matrix is scale^3,
        # but because T = scale * R the Frobenius norm gives us: ||T||_F = scale * sqrt(3)
        # More directly: scale = norm_target / norm_source (same formula as the code)
        norm_src = np.linalg.norm(src)
        norm_tgt = np.linalg.norm(tgt)
        if norm_src > 0:
            scale = norm_tgt / norm_src
        else:
            scale = float("nan")

        scale_factors.append(float(scale))

        angle_rad = _rotation_angle_rad(src, tgt)
        rotation_angles_deg.append(float(np.degrees(angle_rad)))

        if not np.isnan(scale) and (scale < SCALE_LOW or scale > SCALE_HIGH):
            extreme_scale_markers.append(f"{name}={scale:.3f}")

    # ------------------------------------------------------------------
    # Try the full transformation via Animal3D (catches any other failures)
    # ------------------------------------------------------------------
    if transformation_ok:
        try:
            from kinematic_morphospace.cross_species import integrate_dataframe_to_bird3D
            from scipy.linalg import block_diag

            markers_dict = integrate_dataframe_to_bird3D(species_rows, row_idx=row_idx)
            target_bird_3d = Animal3D("hawk", data=markers_dict)
            target_full_markers = target_bird_3d.right_markers.reshape(-1, 3)
            hawk_full_markers = hawk_markers  # already the moving markers

            # The actual transform uses all right_markers, not just the 4 we track above.
            # Attempt the block-diagonal build to catch any additional marker failures.
            T_list = []
            for i in range(len(hawk_full_markers)):
                T_list.append(
                    compute_transformation_matrix(
                        hawk_full_markers[i], target_full_markers[i]
                    )
                )
            block_diag(*T_list)  # will raise if any marker failed

        except ValueError as exc:
            transformation_ok = False
            error_message = str(exc)
        except Exception as exc:  # noqa: BLE001
            transformation_ok = False
            error_message = f"Unexpected error: {exc}"

    # ------------------------------------------------------------------
    # Build summary row
    # ------------------------------------------------------------------
    valid_scales = [s for s in scale_factors if not np.isnan(s)]
    min_scale = min(valid_scales) if valid_scales else float("nan")
    max_scale = max(valid_scales) if valid_scales else float("nan")

    valid_angles = [a for a in rotation_angles_deg if not np.isnan(a)]
    mean_rotation = float(np.mean(valid_angles)) if valid_angles else float("nan")

    # Compute scale variance: high ratio of max/min scale suggests non-uniform deformation
    scale_ratio = (max_scale / min_scale) if (min_scale > 0 and not np.isnan(min_scale)) else float("nan")
    high_scale_variance = not np.isnan(scale_ratio) and scale_ratio > SCALE_VARIANCE_WARN

    # Flag markers with rotation > ROTATION_WARN_DEG
    high_rotation_markers = [
        f"{name}={rotation_angles_deg[i]:.1f}°"
        for i, name in enumerate(MARKER_NAMES)
        if not np.isnan(rotation_angles_deg[i]) and rotation_angles_deg[i] > ROTATION_WARN_DEG
    ]

    issues: list[str] = []
    warnings: list[str] = []

    # Hard failures (FAIL status)
    if n_nan_markers > 0:
        issues.append(f"NaN markers: {', '.join(nan_marker_names)}")
    if n_near_zero > 0:
        issues.append(f"Near-zero markers (<{NEAR_ZERO_THRESHOLD}m): {', '.join(near_zero_names)}")
    if extreme_scale_markers:
        issues.append(f"Extreme scale/rotation: {'; '.join(extreme_scale_markers)}")
    if not transformation_ok and error_message:
        issues.append(f"Transform error: {error_message}")

    # Soft warnings (WARN status — transformation succeeds but output may look wrong)
    if high_scale_variance:
        warnings.append(f"High scale variance: max/min ratio={scale_ratio:.2f}")
    if high_rotation_markers:
        warnings.append(f"High rotation markers (>{ROTATION_WARN_DEG}°): {', '.join(high_rotation_markers)}")

    if issues:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    all_issues = issues + warnings

    return {
        "species": species_name,
        "wingspan_cm": round(float(wingspan_cm), 2) if not np.isnan(float(wingspan_cm)) else float("nan"),
        "n_nan_markers": n_nan_markers,
        "n_near_zero_markers": n_near_zero,
        "min_scale": round(min_scale, 4) if not np.isnan(min_scale) else float("nan"),
        "max_scale": round(max_scale, 4) if not np.isnan(max_scale) else float("nan"),
        "scale_ratio_max_min": round(scale_ratio, 3) if not np.isnan(scale_ratio) else float("nan"),
        "mean_rotation_deg": round(mean_rotation, 2) if not np.isnan(mean_rotation) else float("nan"),
        "status": status,
        "issues": "; ".join(all_issues) if all_issues else "none",
        # Per-marker detail columns for deeper inspection
        **{f"norm_{name}": (round(float(norms[i]), 6) if not np.isnan(norms[i]) else float("nan"))
           for i, name in enumerate(MARKER_NAMES)},
        **{f"scale_{name}": (round(float(scale_factors[i]), 4) if not np.isnan(scale_factors[i]) else float("nan"))
           for i, name in enumerate(MARKER_NAMES)},
        **{f"rotation_deg_{name}": (round(float(rotation_angles_deg[i]), 2) if not np.isnan(rotation_angles_deg[i]) else float("nan"))
           for i, name in enumerate(MARKER_NAMES)},
    }


# ---------------------------------------------------------------------------
# Build the full pipeline (mirroring notebook 14, but including all species)
# ---------------------------------------------------------------------------

def build_pipeline() -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the merged DataFrame.

    This reproduces exactly the steps in notebook 14 (``14_CrossSpeciesGeneralisation.ipynb``)
    but deliberately **omits** the Steller's jay exclusion filter so that known
    problem species are included in the diagnostic output.

    Returns:
        Fully preprocessed merged DataFrame with derived marker columns.
    """
    print("Loading Harvey dataset …")
    wing_df, body_df = load_harvey_data(str(WING_FILE), str(BODY_FILE))

    # Select maximum wingspan row per individual
    wing_df = select_max_wingspan_row(
        wing_df, bird_id_col="BirdID", left_marker="pt8", right_marker="pt12"
    )

    # Clean body data (retain relevant morphological columns)
    body_df = clean_body_data(body_df)

    # NOTE: Steller's jay is intentionally NOT excluded here so we can diagnose it.

    # Extract species and BirdID from compound bird_id strings
    body_df = process_body_bird_id(body_df, id_col="bird_id")

    # Filter wing data to the marker columns used in the transformation
    marker_names = ["pt1", "pt2", "pt4", "pt8", "pt9", "pt10", "pt11"]
    base_columns = ["BirdID", "species"]
    wing_df = filter_marker_columns(wing_df, marker_names, base_columns)

    # Merge wing and body data on BirdID
    merged_df = merge_bird_data(wing_df, body_df, on_col="BirdID")

    print(f"Merged: {len(merged_df)} rows, {merged_df['species_common'].nunique()} species")

    # Recentre coordinates and remap axes (same defaults as notebook 14)
    merged_df = set_new_origin_and_axes(
        merged_df,
        origin_marker=["pt2", "pt11"],
        origin_axes=("x",),
        new_axes=("y", "x", "-z"),
    )

    # Compute derived bilateral markers (wingtip, primary, secondary, etc.)
    merged_df = compute_derived_markers(merged_df)

    # Adjust shoulder distance to match body measurements
    merged_df = check_and_fix_shoulder_distance(merged_df)

    # Correct left/right sign convention
    merged_df = fix_leftright_sign(merged_df)

    return merged_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full diagnostic and print/save results."""
    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    merged_df = build_pipeline()

    # ------------------------------------------------------------------
    # Load hawk reference shape
    # ------------------------------------------------------------------
    print(f"Loading hawk reference shape from {HAWK_SHAPE} …")
    hawk3d = Animal3D("hawk", str(HAWK_SHAPE))
    # Moving right-side markers: shape (n_markers, 3)
    hawk_markers = hawk3d.right_markers.reshape(-1, 3)
    print(f"Hawk right markers: {hawk_markers.shape[0]} markers")

    # ------------------------------------------------------------------
    # Analyse each species
    # ------------------------------------------------------------------
    species_list = sorted(merged_df["species_common"].dropna().unique())
    print(f"\nDiagnosing {len(species_list)} species …\n")

    results: list[dict] = []
    for species_name in species_list:
        species_rows: pd.DataFrame = merged_df[merged_df["species_common"] == species_name].copy()
        result = analyse_species(species_name, species_rows, hawk_markers)
        results.append(result)

    # ------------------------------------------------------------------
    # Build summary DataFrame
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Console output: summary table
    # ------------------------------------------------------------------
    summary_cols = [
        "species", "wingspan_cm", "n_nan_markers", "n_near_zero_markers",
        "min_scale", "max_scale", "scale_ratio_max_min", "mean_rotation_deg",
        "status", "issues",
    ]
    summary_df = results_df[summary_cols].copy()

    print("=" * 100)
    print("CROSS-SPECIES TRANSFORMATION DIAGNOSTICS — SUMMARY")
    print("=" * 100)
    with pd.option_context("display.max_colwidth", 60, "display.width", 120):
        print(summary_df.to_string(index=False))

    print()
    n_pass = (summary_df["status"] == "PASS").sum()
    n_warn = (summary_df["status"] == "WARN").sum()
    n_fail = (summary_df["status"] == "FAIL").sum()
    print(f"Results: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL out of {len(summary_df)} species")

    # ------------------------------------------------------------------
    # Per-marker detail for failed/warned species
    # ------------------------------------------------------------------
    failed = results_df[results_df["status"].isin(["FAIL", "WARN"])]
    if not failed.empty:
        print()
        print("=" * 100)
        print("FAILED / WARNED SPECIES — PER-MARKER DETAIL")
        print("=" * 100)
        detail_cols = ["species"] + [f"norm_{n}" for n in MARKER_NAMES] + \
                      [f"scale_{n}" for n in MARKER_NAMES] + \
                      [f"rotation_deg_{n}" for n in MARKER_NAMES]
        detail_df: pd.DataFrame = failed[detail_cols]  # type: ignore[assignment]
        with pd.option_context("display.max_colwidth", 20, "display.width", 180):
            print(detail_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save full results to CSV
    # ------------------------------------------------------------------
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull diagnostics saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
