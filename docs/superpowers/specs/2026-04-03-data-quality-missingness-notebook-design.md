# Design Spec: NB06b — Data Quality & Missingness

**Date:** 2026-04-03
**Issue:** PCA-paper-6kx (new notebook), PCA-paper-j2q (NB06 cleanup)

## Purpose

A supplementary notebook exploring motion capture data limitations: marker dropout patterns, coordinate density differences between complete and partial frames, and evidence of marker mislabelling. Currently exploratory figures sit in NB06 §8.4d; this notebook gives them a proper home with additional analysis.

## Data

- **559k dataset**: `../../data/raw/labelled_markers_with_missing.npz` — bilateral marker data with NaN for occluded markers. Contains both complete and partial frames; must be separated before comparison.
- **Complete dataset**: `../../data/unilateral/data.npy` — 126k straight-flight frames (all 4 unilateral markers present). Used as the trusted reference.
- **Frame metadata**: `../../data/bilateral/frame_info.csv`
- **Wingspan normalisation**: `../../src/kinematic_morphospace/TotalWingspans.yml`
- **Utilities**: `load_missing_marker_dataset()` from `kinematic_morphospace.null_testing`

## Notebook Structure

### §1 — Introduction

Markdown-only. Brief context:
- Motion capture systems produce incomplete frames when markers are occluded by the bird's body, pass below the capture volume, or are mislabelled by reconstruction software.
- The main PCA analysis (NB02) uses only complete 8-marker frames (~126k). A larger dataset (~559k bilateral frames) includes frames with partial marker visibility.
- This notebook explores what the partial data tells us about capture limitations and potential labelling errors.

### §2 — Dataset Exploration

**Goal:** Load the 559k dataset, cleanly separate complete from partial frames, and characterise the missingness.

Cells:
1. **Imports & data loading** — Load 559k dataset via `load_missing_marker_dataset()`. Convert to unilateral. Scale by wingspan.
2. **Separate complete vs partial** — Identify frames with no NaN (complete) vs any NaN (partial). Report counts. All subsequent "partial" analysis uses only genuinely partial frames.
3. **Per-marker dropout rates** — For each of the 4 unilateral markers (wingtip, primary, secondary, tailtip): count how often it is missing. Bar chart or table.
4. **Co-occurrence of missingness** — Heatmap or table: when marker A is missing, how often is marker B also missing? Shows whether markers tend to drop out together (e.g. distal markers during wing-fold).
5. **Per-hawk/per-year breakdown** — Table of dropout rates split by bird and year, if metadata supports it. Brief — just to flag whether one hawk or capture session is an outlier.

### §3 — Coordinate Densities: Complete vs Partial Frames

**Goal:** Visualise where partial-frame markers sit relative to complete frames in physical space.

Cells (moved from NB06 §8.4d cell 36, extended):
1. **x–z density plots** (lateral vs vertical) — 4 rows (one per marker) × 3 columns (complete | partial | difference). 2D density histograms with shared bins/ranges per marker. Blues for complete, Purples for partial, RdBu_r diverging for difference.
2. **x–y density plots** (lateral vs longitudinal) — Same layout as above but in the x–y plane. This projection shows fore-aft patterns and is important for identifying mislabelling (marker ordering along the wing).
3. **Interpretation** — Markdown cell summarising what the density differences show: partial frames over-represent folded-wing and flapping extremes; complete frames over-represent gliding configurations.

### §4 — Where Are the Other Markers When One Drops Out?

**Goal:** For each marker that is missing, show the positions of the remaining markers. Reveals posture-dependent occlusion.

Cells (moved from NB06 §8.4d cell 38, extended):
1. **x–z dropout position plots** — 4 rows (target marker missing) × 3 columns (complete | dropout | difference). "Complete" = all markers present, showing positions of non-target markers. "Dropout" = target missing but others present.
2. **x–y dropout position plots** — Same layout in x–y plane.
3. **Interpretation** — Markdown: wingtip disappears when wings fold (distal markers tuck behind body); tailtip drops out during extreme tail deflections; dropout is posture-dependent, not random.

### §5 — Mislabelling Detection

**Goal:** Formally identify frames where marker labels are likely swapped.

Cells:
1. **Anatomical ordering test** — Along the wing, markers should follow secondary → primary → wingtip in the lateral (x) direction. For each frame in the partial dataset where multiple wing markers are present, check whether this ordering holds. Count violations. Report rate as percentage of testable frames.
2. **Ordering violation breakdown** — Which pair is most commonly swapped? (e.g. wingtip ↔ primary vs primary ↔ secondary). Table of swap counts.
3. **Spatial evidence** — Overlay the ordering-violation frames on the density plots from §3 to show they cluster in the regions of density mismatch (the red blobs in the difference plots). This ties the formal test back to the visual evidence.
4. **Interpretation** — Markdown: mislabelling rate, which markers are most affected, likely cause (reconstruction software confusing nearby markers during rapid movement), implications for analysis (the complete dataset used for PCA excludes these frames by construction, since mislabelled frames would fail the 8-marker completeness check or produce unphysical configurations).

### §6 — Summary & Limitations

Markdown-only. Honest accounting:
- Motion capture dropout is posture-dependent, not random — the complete dataset under-represents extreme wing configurations.
- Mislabelling occurs at a quantifiable rate, primarily between adjacent markers.
- Ghost markers (spurious reflections) are a known issue but not formally tested here.
- These limitations affect sampling density, not the morphospace structure itself (cross-reference NB06 robustness tests).

## Figures Output

Save to `../../figures/supplementary/`:
- `S06b_density_xz.pdf` — §3 x–z density comparison
- `S06b_density_xy.pdf` — §3 x–y density comparison
- `S06b_dropout_positions_xz.pdf` — §4 x–z dropout positions
- `S06b_dropout_positions_xy.pdf` — §4 x–y dropout positions
- `S06b_ordering_violations.pdf` — §5 spatial evidence overlay

(Figure naming is provisional — will align with final supplementary numbering.)

## Relationship to NB06

After this notebook is created:
- **Move from NB06 §8.4d**: Raw density heatmaps (cell 36) and dropout position plots (cell 38). These become §3 and §4 here, extended with x–y projections.
- **Keep in NB06 §8.4d**: The coverage comparison (grey/blue/red bin plot, cell 37) stays as the quantitative robustness answer. Per PCA-paper-j2q.
- NB06 §8.4d will cross-reference this notebook for the detailed exploration.

## Out of Scope

- Correcting mislabelled frames (we detect, not fix)
- Ghost marker detection (mentioned in summary only)
- Changes to the main PCA pipeline
