"""Display helpers for robustness-validation tables.

Each function accepts pre-computed values and prints a neatly
aligned table.  They are intentionally one-off formatters — kept
outside the notebook so cells show only the computation.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# ── §8.2 CEV comparison across representations ────────────────────────

def print_cev_comparison(
    rows: Sequence[tuple[str, np.ndarray]],
    n_modes: int = 4,
) -> None:
    """Print a method x cumulative-explained-variance table.

    Args:
        rows: Sequence of (method_name, cev_array) tuples. Each
            ``cev_array`` must have at least ``n_modes`` entries.
        n_modes: Number of cumulative-explained-variance columns to show.
            Defaults to 4.
    """
    header = f"{'Method':>25s}"
    for m in range(1, n_modes + 1):
        header += f" {'CEV' + chr(0x2080 + m):>8s}"
    print(header)
    print("─" * (25 + 9 * n_modes))
    for name, cev in rows:
        vals = "".join(f" {cev[m]:>8.4f}" for m in range(n_modes))
        print(f"{name:>25s}{vals}")


# ── §8.4c Occlusion-conditioned PCA ───────────────────────────────────

def print_occlusion_pca(
    rows: Sequence[dict],
) -> None:
    """Print a marker-occlusion PCA comparison table.

    Args:
        rows: Sequence of dicts with keys ``marker``, ``n_occluded``,
            ``cev_complete``, ``cev_occluded``, and ``cosines`` (array).
            Set ``cosines=None`` when ``n_occluded`` is too small to
            compute a meaningful cosine.
    """
    n_cos = max(
        (len(r["cosines"]) for r in rows if r["cosines"] is not None),
        default=4,
    )
    cos_hdr = "  ".join(f"{i + 1:>5d}" for i in range(n_cos))

    print(
        f"{'Dropped':<12s} {'Occluded':>10s}   {'── CEV₄ ──':^19s}   "
        f"{'Cosines (modes 1-' + str(n_cos) + ')':^{6 * n_cos}s}"
    )
    print(
        f"{'marker':<12s} {'frames':>10s}   {'Complete':>9s} {'Occluded':>9s}   "
        f"{cos_hdr}"
    )
    print("─" * (24 + 20 + 6 * n_cos))

    for r in rows:
        if r["cosines"] is None:
            print(f"{r['marker']:<12s} {r['n_occluded']:>10,}   — too few frames")
        else:
            cos_str = "  ".join(f"{c:>5.3f}" for c in r["cosines"])
            print(
                f"{r['marker']:<12s} {r['n_occluded']:>10,}   "
                f"{r['cev_complete']:>9.4f} {r['cev_occluded']:>9.4f}   "
                f"{cos_str}"
            )


# ── §8.4d Least-squares projection validation ────────────────────────

def print_projection_validation(
    n_frames: int,
    pc1_sd: float,
    pc2_sd: float,
    rows: Sequence[dict],
) -> None:
    """Print a projection-validation table for single and double marker masking.

    Args:
        n_frames: Total number of frames used (printed in the header).
        pc1_sd: Standard deviation of PC1 scores (printed in the header).
        pc2_sd: Standard deviation of PC2 scores (printed in the header).
        rows: Sequence of dicts with keys ``label``, ``rmse_pc1``,
            ``rmse_pc2``, ``corr_pc1``, ``corr_pc2``. Insert a dict
            ``{"separator": True}`` to emit a blank line between
            single- and double-marker blocks.
    """
    print("Least-squares projection validation")
    print(f"Frames: {n_frames:,}  |  PC1 SD: {pc1_sd:.4f}  |  PC2 SD: {pc2_sd:.4f}")
    print()
    print(
        f"{'Masked':<24s} {'PC1 RMSE':>9s} {'PC2 RMSE':>9s} "
        f"{'PC1 r':>7s} {'PC2 r':>7s}"
    )
    print("─" * 55)

    for r in rows:
        if r.get("separator"):
            print()
            continue
        print(
            f"{r['label']:<24s} {r['rmse_pc1']:>9.4f} {r['rmse_pc2']:>9.4f} "
            f"{r['corr_pc1']:>7.4f} {r['corr_pc2']:>7.4f}"
        )


# ── §8.5 Temporal autocorrelation & thinned PCA ──────────────────────

def print_thinned_pca(
    rows: Sequence[dict],
    n_components: int = 4,
) -> None:
    """Print a thinned-PCA stability table across temporal subsampling steps.

    Args:
        rows: Sequence of dicts with keys ``step``, ``n_frames``, ``cev``,
            ``variance_per_mode`` (array), and ``cosines`` (array).
        n_components: Number of modes to show per row. Defaults to 4.
    """
    pc_hdr = "".join(f" {'PC' + str(i + 1):>7s}" for i in range(n_components))
    cosines_label = f"Cosines (1-{n_components})"
    cosines_width = 5 * n_components + n_components - 1
    print(
        f"{'Step':>5s} {'Frames':>10s} {'CEV₄':>8s}"
        f"{pc_hdr}   {cosines_label:>{cosines_width}s}"
    )
    print("─" * (25 + 8 * n_components + 5 * n_components + n_components))

    for r in rows:
        var_str = "".join(
            f" {r['variance_per_mode'][i]:>7.4f}" for i in range(n_components)
        )
        cos_str = " ".join(f"{c:.3f}" for c in r["cosines"])
        print(
            f"{r['step']:>5d} {r['n_frames']:>10,} {r['cev']:>8.4f}"
            f"{var_str}   {cos_str}"
        )


# ── §8.6b Bootstrap stability ─────────────────────────────────────────

def print_bootstrap_stability(
    rows: Sequence[dict],
    observed_cev4: float | None = None,
) -> None:
    """Print a bootstrap CEV₄ and cosine stability summary.

    Args:
        rows: Sequence of dicts with keys ``label``, ``n_bootstraps``,
            ``cev4`` (array of bootstrap CEV₄ values), and ``cosines``
            (n_bootstraps x k array of principal cosines).
        observed_cev4: If provided, the observed (non-bootstrap) CEV₄
            value is printed before the table. Defaults to None.
    """
    if observed_cev4 is not None:
        print(f"Observed CEV₄: {observed_cev4:.4f}")
        print()

    print(
        f"{'Bootstrap scheme':<30s} {'CEV₄ mean':>10s} {'± SD':>8s} "
        f"{'95% CI':>17s}   {'Cosines (median)':>23s}   {'Cosines (min)':>23s}"
    )
    print("─" * 118)

    for r in rows:
        cev = r["cev4"]
        cos = r["cosines"]
        ci_lo, ci_hi = np.percentile(cev, 2.5), np.percentile(cev, 97.5)
        cos_med = " ".join(f"{c:.4f}" for c in np.median(cos, axis=0))
        cos_min = " ".join(f"{c:.4f}" for c in np.min(cos, axis=0))
        print(
            f"{r['label']:<30s} {cev.mean():>10.4f} {cev.std():>8.4f} "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]   {cos_med:>23s}   {cos_min:>23s}"
        )


# ── §8.6a Residual eigenvalue structure ───────────────────────────────

def print_residual_eigenvalues(
    residual_variance: np.ndarray,
    shuffled_variance: np.ndarray,
    n_show: int = 8,
) -> None:
    """Print residual vs. shuffled eigenvalue comparison.

    Args:
        residual_variance: Per-component variance remaining after removing
            the leading modes.
        shuffled_variance: Per-component variance from column-shuffled data,
            representing the null distribution.
        n_show: Number of components to display. Defaults to 8.
    """
    print("Residual eigenvalue structure (after removing 4 modes):")
    print(f"{'Component':>10s} {'Residual':>14s} {'Shuffled':>14s} {'Ratio':>8s}")
    print("─" * 50)
    for comp in range(n_show):
        ratio = (
            residual_variance[comp] / shuffled_variance[comp]
            if shuffled_variance[comp] > 0
            else float("inf")
        )
        print(
            f"{comp + 1:>10d} {residual_variance[comp]:>14.6f} "
            f"{shuffled_variance[comp]:>14.6f} {ratio:>8.2f}x"
        )


# ── §8.6a Reconstruction RMSE by quintile ────────────────────────────

def print_quintile_rmse(
    all_scores: np.ndarray,
    frame_rmse: np.ndarray,
    pc_pairs: Sequence[tuple[int, str]] = (
        (0, "PC1 (wing lifting)"),
        (1, "PC2 (wing spreading)"),
    ),
) -> None:
    """Print reconstruction RMSE broken down by PC-score quintile.

    Args:
        all_scores: PCA score array of shape ``(n_frames, n_components)``.
        frame_rmse: Per-frame reconstruction RMSE array of shape ``(n_frames,)``.
        pc_pairs: Sequence of (component_index, display_name) tuples
            specifying which PCs to break down. Defaults to PC1 and PC2.
    """
    print("Reconstruction RMSE by PC score quintile:")
    for pc_index, pc_name in pc_pairs:
        quintile_labels = pd.qcut(all_scores[:, pc_index], 5, labels=False)
        print(f"\n  {pc_name}:")
        for quintile in range(5):
            in_quintile = quintile_labels == quintile
            mean_score = all_scores[in_quintile, pc_index].mean()
            mean_rmse = frame_rmse[in_quintile].mean()
            print(
                f"    Q{quintile + 1} (score={mean_score:>7.3f}): RMSE = "
                f"{mean_rmse:.4f}"
            )


# ── §8.6a Local PCA stability ────────────────────────────────────────

def print_local_pca_stability(
    rows: Sequence[dict],
    n_modes: int = 4,
) -> None:
    """Print a local-PCA cosine table stratified by PC1 quintile.

    Args:
        rows: Sequence of dicts with keys ``label`` (quintile name) and
            ``cosines`` (array of principal cosines for each mode).
        n_modes: Number of modes to display. Defaults to 4.
    """
    cos_hdr = "".join(f" {'cos' + str(i + 1):>6s}" for i in range(n_modes))
    print("Local PCA stability (splitting data by PC1 quintile):")
    print(f"{'Quintile':>10s}{cos_hdr}")
    print("─" * (10 + 7 * n_modes))

    for r in rows:
        cos_str = "".join(f" {r['cosines'][i]:>6.3f}" for i in range(n_modes))
        print(f"{r['label']:>10s}{cos_str}")


# ── §8.6a Intrinsic dimensionality ───────────────────────────────────

def print_intrinsic_dimensionality(
    rows: Sequence[dict],
) -> None:
    """Print a Levina-Bickel intrinsic-dimensionality table.

    Args:
        rows: Sequence of dicts with keys ``n_neighbours``, ``median_id``,
            and ``mean_id``.
    """
    print("Intrinsic dimensionality (Levina-Bickel, 5 000-point sample):")
    print(f"{'Neighbours':>12s} {'Median ID':>10s} {'Mean ID':>10s}")
    print("─" * 35)
    for r in rows:
        print(f"{r['n_neighbours']:>12d} {r['median_id']:>10.1f} {r['mean_id']:>10.1f}")


# ── §13 Flight-behaviour continuum (NB12) ─────────────────────────────

def print_flight_phase_trace_summary(traces) -> None:
    """Print per-flight-phase PC1/PC2 within-bin standard deviation.

    Args:
        traces: Dict mapping phase name to a DataFrame (or None). Each
            DataFrame is indexed by distance-bin midpoint and has columns
            ``PC1_std`` and ``PC2_std``. ``None`` values are skipped.
    """
    rows = []
    for name, g in traces.items():
        if g is None or len(g) == 0:
            continue
        d_lo, d_mid, d_hi = g.index.min(), g.index[len(g) // 2], g.index.max()
        for d in [d_lo, d_mid, d_hi]:
            row = g.loc[d]
            rows.append((d, name, row["PC1_std"], row["PC2_std"]))

    rows.sort(key=lambda r: -r[0])  # far → near (descending distance)

    print("Within-bin PC1 std at representative distances (9m flights):")
    print(f"{'distance (m)':>14} {'phase':>18} {'PC1 std':>9} {'PC2 std':>9}")
    for d, name, pc1_std, pc2_std in rows:
        print(f"{d:>14.2f} {name:>18} {pc1_std:>9.3f} {pc2_std:>9.3f}")


def print_overlap_metrics(
    *,
    n_dims: int,
    n_flap: int,
    n_glide: int,
    silhouette: float,
    centroid_dist: float,
    within_spread: float,
    mahalanobis: float,
    lda_mean: float,
    lda_std: float,
    class_prior: float,
) -> None:
    """Print a silhouette / Mahalanobis / LDA overlap summary for flapping vs gliding.

    Args:
        n_dims: Number of PCA dimensions used for the analysis.
        n_flap: Number of flapping frames.
        n_glide: Number of gliding frames.
        silhouette: Silhouette score using a priori labels.
        centroid_dist: Euclidean distance between class centroids.
        within_spread: Square root of mean within-class variance.
        mahalanobis: Mahalanobis distance between centroids.
        lda_mean: Mean LDA 5-fold cross-validation accuracy.
        lda_std: Standard deviation of LDA cross-validation accuracy.
        class_prior: Always-predict-majority-class accuracy (baseline).
    """
    print(f"Flapping vs gliding in {n_dims}-D score space")
    print(f"  n (flapping): {n_flap:,}")
    print(f"  n (gliding):  {n_glide:,}")
    print()
    print(
        f"Silhouette (a priori labels):        {silhouette:+.3f}   "
        f"(> 0.5 = distinct, ~0 = continuum)"
    )
    print(f"Centroid Euclidean distance:         {centroid_dist:.3f}")
    print(f"Within-class spread (sqrt mean var): {within_spread:.3f}")
    print(
        f"Mahalanobis centroid separation:     {mahalanobis:.2f}   "
        f"(>> 1 = separated)"
    )
    print()
    print(f"LDA 5-fold CV accuracy:              {lda_mean:.3f} ± {lda_std:.3f}")
    print(f"Class prior (always-guess):          {class_prior:.3f}")
    print(f"Improvement over prior:              {lda_mean - class_prior:+.3f}")


# ── §15 Alternative method comparison ───────────────────────────────────

def print_method_comparison(
    method_results: dict,
    notes_map: dict | None = None,
) -> None:
    """Print a formatted comparison table of dimensionality reduction methods.

    Args:
        method_results: Mapping of method name → dict with keys
            ``min_cosine_4`` (float) and ``rmse_4`` (float).
        notes_map: Optional mapping of method name → notes string. If
            omitted, all notes columns are left blank. Defaults to None.
    """
    if notes_map is None:
        notes_map = {}

    print(f"\n{'Method':>25} {'Min cos(k=4)':>14} {'RMSE(k=4)':>12} {'Notes':>30}")
    print("-" * 85)
    for name, r in method_results.items():
        notes = notes_map.get(name, "")
        print(f"{name:>25} {r['min_cosine_4']:>14.4f} {r['rmse_4']:>12.6f} {notes:>30}")


def print_bic_summary(bic_curve, best_k: int) -> None:
    """Print BIC at k=2, at the BIC minimum, and the decrease between them.

    Args:
        bic_curve: Array of BIC values indexed by k-1 (i.e. bic_curve[0]
            corresponds to k=1).
        best_k: The cluster count at which BIC is minimised.
    """
    print(f"BIC at k=2:   {bic_curve[1]:,.0f}")
    print(f"BIC at k={best_k}:  {bic_curve[best_k - 1]:,.0f}")
    print(
        f"BIC decrease from k=2 to k={best_k}: "
        f"{bic_curve[1] - bic_curve[best_k - 1]:,.0f}"
    )


# ── NB07 Missingness and Sampling Bias ───────────────────────────────────

def print_complete_markers_summary(
    n_complete: int,
    complete_markers: np.ndarray,
) -> None:
    """Print shape summary for the complete-marker dataset.

    Args:
        n_complete: Number of unilateral frames with all markers present.
        complete_markers: Complete-marker array of shape
            ``(N, n_markers, 3)`` used for shape metadata only.
    """
    print(f"Complete markers: {n_complete:,} unilateral frames")
    print(f"  ({n_complete // 2:,} bilateral x 2 sides)")
    print(f"  Markers per frame: {complete_markers.shape[1]}")
    print(f"  Coordinates: {complete_markers.shape[2]} (x, y, z)")
    print()


def print_partial_markers_summary(
    partial_bilateral: np.ndarray,
    partial_unilateral: np.ndarray,
    complete_unilateral: np.ndarray,
) -> None:
    """Print frame-count summary for the partial-marker dataset.

    Args:
        partial_bilateral: Bilateral partial-marker array (≥1 NaN marker).
        partial_unilateral: Unilateral partial-marker array restricted to
            straight flight.
        complete_unilateral: Complete straight-flight unilateral frames
            used as the comparison count.
    """
    print(f"Partial markers: {partial_bilateral.shape[0]:,} bilateral frames")
    print(f"  Straight-flight unilateral: {partial_unilateral.shape[0]:,}")
    print(f"  Complete straight-flight unilateral: {complete_unilateral.shape[0]:,}")


def print_dataset_split(
    complete_data: np.ndarray,
    partial_data: np.ndarray,
    partial_unilateral: np.ndarray,
) -> None:
    """Print complete-vs-partial frame counts and the partial fraction.

    Args:
        complete_data: Frames with all markers present (no NaN).
        partial_data: Frames with at least one missing marker.
        partial_unilateral: Full unilateral dataset combining complete
            and partial frames, used for computing the partial fraction.
    """
    print("Within the broader dataset:")
    print(f"  Complete frames (all 4 markers): {complete_data.shape[0]:,}")
    print(f"  Partial frames (≥1 marker missing): {partial_data.shape[0]:,}")
    partial_fraction = partial_data.shape[0] / partial_unilateral.shape[0]
    print(f"  Partial fraction: {partial_fraction:.1%}")


def print_density_shift_table(
    marker_names: Sequence[str],
    complete_data: np.ndarray,
    partial_data: np.ndarray,
    bins: int = 60,
) -> None:
    """Print per-marker density-shift between complete and partial frames.

    For each marker, bins positions using the complete-frame distribution and
    reports what fraction of each group falls in the densest 25 % of bins.
    A negative shift indicates that partial frames are under-represented
    in the high-density (spread/gliding) region relative to complete frames.

    Args:
        marker_names: Names for each marker column.
        complete_data: Complete (no-NaN) frames of shape
            ``(N_complete, n_markers, 3)``.
        partial_data: Partial (≥1 NaN) frames of shape
            ``(N_partial, n_markers, 3)``.
        bins: Number of histogram bins per axis. Defaults to 60.
    """
    print(
        f'{"Marker":<12} {"Complete in densest 25%":>24} '
        f'{"Partial in densest 25%":>24} {"Shift":>8}'
    )
    print('-' * 72)

    for m, name in enumerate(marker_names):
        xc = complete_data[:, m, 0]
        zc = complete_data[:, m, 2]

        present = ~np.isnan(partial_data[:, m, 0])
        xp = partial_data[present, m, 0]
        zp = partial_data[present, m, 2]

        x_range = (min(xc.min(), xp.min()), max(xc.max(), xp.max()))
        z_range = (min(zc.min(), zp.min()), max(zc.max(), zp.max()))

        H_c, xedges, zedges = np.histogram2d(
            xc, zc, bins=bins, range=[x_range, z_range]
        )
        occupied = H_c[H_c > 0]
        threshold = np.percentile(occupied, 75)
        dense = H_c >= threshold

        def _bin_fraction(
            x: np.ndarray, z: np.ndarray, mask_2d: np.ndarray,
            _xe: np.ndarray = xedges, _ze: np.ndarray = zedges,
        ) -> float:
            xi = np.clip(np.searchsorted(_xe, x) - 1, 0, bins - 1)
            zi = np.clip(np.searchsorted(_ze, z) - 1, 0, bins - 1)
            return mask_2d[xi, zi].mean()

        frac_c = _bin_fraction(xc, zc, dense)
        frac_p = _bin_fraction(xp, zp, dense)

        print(f'{name:<12} {frac_c:>23.1%} {frac_p:>23.1%} {frac_p - frac_c:>+7.1f}pp')

    print()
    print('Densest 25% of bins = gliding/spread-wing region (most complete frames).')
    print('Negative shift = partial frames are less concentrated at gliding peak.')


def print_marker_dropout_rates(
    marker_names: Sequence[str],
    partial_data: np.ndarray,
) -> None:
    """Print per-marker dropout rates within partial frames.

    Args:
        marker_names: Names for each marker column.
        partial_data: Frames containing at least one missing marker,
            of shape ``(N_partial, n_markers, 3)``.
    """
    n_partial = partial_data.shape[0]
    print('Marker dropout rates in partial frames:')
    print(f'{"Marker":<12} {"Missing":>10} {"Present":>10} {"Dropout %":>10}')
    print('-' * 45)
    for m, name in enumerate(marker_names):
        missing_count = np.isnan(partial_data[:, m, 0]).sum()
        present_count = n_partial - missing_count
        dropout_rate = missing_count / n_partial
        print(
            f'{name:<12} {missing_count:>10,} {present_count:>10,} '
            f'{dropout_rate:>10.1%}'
        )


def print_anatomical_violations(
    all_data: np.ndarray,
    pairs: Sequence[tuple[str, str, int, int]],
) -> dict[str, np.ndarray]:
    """Print anatomical ordering violations and return per-pair boolean masks.

    Tests whether the expected lateral x-ordering holds for each marker pair
    (inner/medial marker should have a smaller x value than outer/lateral).

    Args:
        all_data: All unilateral frames of shape ``(N, n_markers, 3)``.
        pairs: Sequence of (name_inner, name_outer, idx_inner, idx_outer)
            tuples naming the expected inner (medial) and outer (lateral)
            markers and giving their column indices.

    Returns:
        Dict mapping pair keys (``'name_inner_vs_name_outer'``) to full-length
        boolean arrays where True indicates a violation frame.
    """
    print(f'{"Pair":<25} {"Testable":>10} {"Violations":>12} {"Rate":>8}')
    print('-' * 58)

    violation_masks: dict[str, np.ndarray] = {}
    for name_inner, name_outer, idx_inner, idx_outer in pairs:
        inner_present = ~np.isnan(all_data[:, idx_inner, 0])
        outer_present = ~np.isnan(all_data[:, idx_outer, 0])
        testable = inner_present & outer_present

        x_inner = all_data[testable, idx_inner, 0]
        x_outer = all_data[testable, idx_outer, 0]
        violated = x_inner >= x_outer

        pair_name = f'{name_inner} >= {name_outer}'
        n_testable = testable.sum()
        n_violated = violated.sum()
        violation_rate = n_violated / n_testable
        print(
            f'{pair_name:<25} {n_testable:>10,} {n_violated:>12,} '
            f'{violation_rate:>8.2%}'
        )

        full_mask = np.zeros(all_data.shape[0], dtype=bool)
        full_mask[testable] = violated
        violation_masks[f'{name_inner}_vs_{name_outer}'] = full_mask

    return violation_masks


def print_violation_breakdown(
    violation_masks: dict[str, np.ndarray],
    any_nan: np.ndarray,
) -> None:
    """Print violation counts split by complete vs partial frames.

    Args:
        violation_masks: Per-pair boolean masks from
            :func:`print_anatomical_violations`.
        any_nan: Boolean array of shape ``(N,)`` where True indicates a
            frame with at least one missing marker.
    """
    print(f'{"Pair":<25} {"Complete":>12} {"Partial":>12}')
    print('-' * 52)
    for key, mask in violation_masks.items():
        n_complete_viol = mask[~any_nan].sum()
        n_partial_viol = mask[any_nan].sum()
        n_complete_total = (~any_nan).sum()
        n_partial_total = any_nan.sum()
        complete_rate = n_complete_viol / n_complete_total
        partial_rate = n_partial_viol / n_partial_total
        print(
            f'{key:<25} {n_complete_viol:>6,} ({complete_rate:.2%}) '
            f'{n_partial_viol:>6,} ({partial_rate:.2%})'
        )
