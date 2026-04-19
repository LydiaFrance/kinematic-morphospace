"""Plotting functions for missingness and sampling bias analysis."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_density_comparison(
    complete_data: np.ndarray,
    partial_data: np.ndarray,
    marker_names: list[str],
    coord_a: int,
    coord_b: int,
    label_a: str,
    label_b: str,
    suptitle: str,
    bins: int = 60,
):
    """Plot 2D density comparison: complete vs partial vs difference.

    Creates one column of three panels per marker, stacked vertically.
    Row 0 shows where complete frames sit in position space, row 1 shows the
    same for partial frames (only frames where this marker is visible), and
    row 2 shows the difference (red = partial frames more common, blue = less).

    Args:
        complete_data: Array of shape (N_complete, n_markers, 3) with frames
            where all four markers were successfully tracked.
        partial_data: Array of shape (N_partial, n_markers, 3) with frames
            where at least one marker is missing (NaN).
        marker_names: Labels for each marker — one column of panels per name.
        coord_a: Which spatial coordinate for the horizontal axis.
            0 = x (lateral), 1 = y (longitudinal), 2 = z (vertical).
        coord_b: Which spatial coordinate for the vertical axis.
        label_a: Axis label for the horizontal plot axis.
        label_b: Axis label for the vertical plot axis.
        suptitle: Title printed above all panels.
        bins: Number of histogram bins per axis (default 60).

    Returns:
        matplotlib Figure object.
    """
    n_markers = len(marker_names)
    fig, axes = plt.subplots(
        n_markers, 3,
        figsize=(9.5, 3.2 * n_markers),
        constrained_layout=True,
    )

    col_labels = ['Complete\n(all markers)', 'Partial\n(marker present,\nothers missing)', 'Partial - Complete']
    shared_range = [(-0.1, 0.55), (-0.55, 0.45)]

    for m, name in enumerate(marker_names):
        ac = complete_data[:, m, coord_a]
        bc = complete_data[:, m, coord_b]

        present = ~np.isnan(partial_data[:, m, 0])
        ap = partial_data[present, m, coord_a]
        bp = partial_data[present, m, coord_b]

        H_c, aedges, bedges = np.histogram2d(ac, bc, bins=bins, range=shared_range, density=True)
        H_p, _, _ = np.histogram2d(ap, bp, bins=bins, range=shared_range, density=True)

        vmax = np.nanquantile(np.concatenate([H_c.ravel(), H_p.ravel()]), 0.95)

        # Col 0: complete frames
        H_plot = H_c.T.copy()
        H_plot[H_plot == 0] = np.nan
        im = axes[m, 0].pcolormesh(aedges, bedges, H_plot, cmap='Blues', vmax=vmax)
        axes[m, 0].set_ylabel(f'{name}\n{label_b}', fontweight='bold')
        plt.colorbar(im, ax=axes[m, 0])

        # Col 1: partial frames
        H_plot = H_p.T.copy()
        H_plot[H_plot == 0] = np.nan
        im = axes[m, 1].pcolormesh(aedges, bedges, H_plot, cmap='Purples', vmax=vmax)
        plt.colorbar(im, ax=axes[m, 1])

        # Col 2: difference map
        H_diff = H_p.T - H_c.T
        mask = (H_c.T > 0) | (H_p.T > 0)
        H_diff[~mask] = np.nan
        vabs = np.nanquantile(np.abs(H_diff), 0.98)
        im = axes[m, 2].pcolormesh(
            aedges, bedges, H_diff, cmap='RdBu_r', vmin=-vabs, vmax=vabs
        )
        plt.colorbar(im, ax=axes[m, 2], label='Δ Density')

        for c in range(3):
            axes[m, c].set_aspect('equal')

    # X-axis labels on bottom row only
    for c in range(3):
        axes[-1, c].set_xlabel(label_a)
        axes[0, c].set_title(col_labels[c])

    fig.suptitle(suptitle)
    return fig


def plot_dropout_positions(
    complete_data: np.ndarray,
    partial_data: np.ndarray,
    marker_names: list[str],
    coord_a: int,
    coord_b: int,
    label_a: str,
    label_b: str,
    suptitle: str,
    bins: int = 60,
):
    """For each target marker that is missing, plot positions of the other markers.

    One column per target marker, with three comparison panels
    (complete, dropout, difference) stacked vertically.

    Args:
        complete_data: Array of shape (N_complete, n_markers, 3) with frames
            where all four markers were successfully tracked.
        partial_data: Array of shape (N_partial, n_markers, 3) with frames
            where at least one marker is missing (NaN).
        marker_names: Labels for each marker — one column of panels per name.
        coord_a: Which spatial coordinate for the horizontal axis.
        coord_b: Which spatial coordinate for the vertical axis.
        label_a: Axis label for the horizontal plot axis.
        label_b: Axis label for the vertical plot axis.
        suptitle: Title printed above all panels.
        bins: Number of histogram bins per axis (default 60).

    Returns:
        matplotlib Figure object.
    """
    n_markers = len(marker_names)
    fig, axes = plt.subplots(
        n_markers, 3,
        figsize=(9.5, 3.2 * n_markers),
        constrained_layout=True,
    )

    col_labels = ['Other markers\n(complete)', 'Other markers\n(dropout)', 'Dropout - Complete']
    shared_range = [(-0.1, 0.55), (-0.55, 0.45)]

    for m_target, target_name in enumerate(marker_names):
        others = [i for i in range(len(marker_names)) if i != m_target]

        ac = complete_data[:, others, coord_a].ravel()
        bc = complete_data[:, others, coord_b].ravel()

        target_missing = np.isnan(partial_data[:, m_target, 0])
        others_present = np.all(~np.isnan(partial_data[:, others, 0]), axis=1)
        dropout_mask = target_missing & others_present

        ap = partial_data[dropout_mask][:, others, coord_a].ravel()
        bp = partial_data[dropout_mask][:, others, coord_b].ravel()

        n_dropout = dropout_mask.sum()
        print(f'{target_name} missing (others present): {n_dropout:,} frames')

        if n_dropout < 100:
            for c in range(3):
                axes[m_target, c].text(
                    0.5, 0.5, 'Too few frames',
                    transform=axes[m_target, c].transAxes, ha='center'
                )
            continue

        H_c, aedges, bedges = np.histogram2d(ac, bc, bins=bins, range=shared_range, density=True)
        H_p, _, _ = np.histogram2d(ap, bp, bins=bins, range=shared_range, density=True)

        vmax = np.nanquantile(np.concatenate([H_c.ravel(), H_p.ravel()]), 0.95)

        # Col 0: complete (baseline)
        H_plot = H_c.T.copy()
        H_plot[H_plot == 0] = np.nan
        im = axes[m_target, 0].pcolormesh(
            aedges, bedges, H_plot, cmap='Blues', vmax=vmax
        )
        axes[m_target, 0].set_ylabel(f'{target_name}\n{label_b}', fontweight='bold')
        plt.colorbar(im, ax=axes[m_target, 0])

        # Col 1: dropout frames
        H_plot = H_p.T.copy()
        H_plot[H_plot == 0] = np.nan
        im = axes[m_target, 1].pcolormesh(
            aedges, bedges, H_plot, cmap='Purples', vmax=vmax
        )
        plt.colorbar(im, ax=axes[m_target, 1])

        # Col 2: difference
        H_diff = H_p.T - H_c.T
        mask = (H_c.T > 0) | (H_p.T > 0)
        H_diff[~mask] = np.nan
        vabs = np.nanquantile(np.abs(H_diff), 0.98)
        im = axes[m_target, 2].pcolormesh(
            aedges, bedges, H_diff, cmap='RdBu_r', vmin=-vabs, vmax=vabs
        )
        plt.colorbar(im, ax=axes[m_target, 2], label='Δ Density')

        for c in range(3):
            axes[m_target, c].set_aspect('equal')

    # X-axis labels on bottom row only, column titles on top
    for c in range(3):
        axes[-1, c].set_xlabel(label_a)
        axes[0, c].set_title(col_labels[c])

    fig.suptitle(suptitle)
    return fig


def plot_coverage_comparison(
    complete_data: np.ndarray,
    partial_data: np.ndarray,
    marker_names: list[str],
    bins: int = 60,
):
    """Plot coverage comparison: shared, complete-only, and novel (partial-only) bins.

    For each marker, bins the x-z positions and classifies each bin as shared
    (both datasets), complete-only, or novel (partial-only).

    Args:
        complete_data: Array of shape (N_complete, n_markers, 3) with frames
            where all markers were successfully tracked.
        partial_data: Array of shape (N_partial, n_markers, 3) with frames
            where at least one marker is missing (NaN).
        marker_names: Labels for each marker — one panel per marker.
        bins: Number of histogram bins per axis (default 60).

    Returns:
        Tuple of (Figure, list of coverage stats dicts).
    """
    n_markers = len(marker_names)
    fig, axes = plt.subplots(1, n_markers, figsize=(4.5 * n_markers, 4.5))
    if n_markers == 1:
        axes = [axes]

    coverage_stats = []

    for m, (name, ax) in enumerate(zip(marker_names, axes)):
        xc = complete_data[:, m, 0]
        zc = complete_data[:, m, 2]

        present = ~np.isnan(partial_data[:, m, 0])
        xp = partial_data[present, m, 0]
        zp = partial_data[present, m, 2]

        x_range = (min(xc.min(), xp.min()), max(xc.max(), xp.max()))
        z_range = (min(zc.min(), zp.min()), max(zc.max(), zp.max()))
        hist_range = [x_range, z_range]

        H_c, xedges, zedges = np.histogram2d(xc, zc, bins=bins, range=hist_range)
        H_p, _, _ = np.histogram2d(xp, zp, bins=bins, range=hist_range)

        both = (H_c > 0) & (H_p > 0)
        complete_only = (H_c > 0) & (H_p == 0)
        novel = (H_c == 0) & (H_p > 0)

        rgb = np.ones((*H_c.T.shape, 4))
        rgb[both.T] = [0.75, 0.85, 0.93, 1.0]
        rgb[complete_only.T] = [0.20, 0.50, 0.55, 1.0]
        rgb[novel.T] = [0.90, 0.40, 0.35, 1.0]

        ax.imshow(
            rgb, origin='lower',
            extent=[x_range[0], x_range[1], z_range[0], z_range[1]],
            aspect='equal',
        )
        # Only set the x label on the bottom row to avoid clutter
        if m == 0:
            ax.set_ylabel('z (vertical)')
        ax.set_title(name, fontweight='bold')

        n_shared = int(both.sum())
        n_complete_only = int(complete_only.sum())
        n_novel = int(novel.sum())
        n_occupied = n_shared + n_complete_only + n_novel
        frames_novel = H_p[novel].sum()
        frames_partial_total = H_p[H_p > 0].sum()
        coverage_stats.append({
            'Marker': name,
            'Shared': n_shared,
            'Complete only': n_complete_only,
            'Novel (partial)': n_novel,
            'Novel % bins': f'{n_novel / n_occupied:.1%}' if n_occupied > 0 else '0%',
            'Frames in novel': f'{frames_novel:,.0f}',
            '% of partial': f'{frames_novel / frames_partial_total:.2%}' if frames_partial_total > 0 else '0%',
        })

    fig.legend(
        [
            Patch(facecolor=[0.75, 0.85, 0.93], edgecolor='0.5'),
            Patch(facecolor=[0.20, 0.50, 0.55], edgecolor='0.3'),
            Patch(facecolor=[0.90, 0.40, 0.35], edgecolor='0.3'),
        ],
        ['Both datasets', 'Complete only', 'Partial only (novel)'],
        loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02),
        frameon=False, fontsize=10,
    )
    fig.suptitle('Coverage comparison: do partial frames visit new regions?', y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig, coverage_stats


def plot_cooccurrence_heatmap(
    partial_data: np.ndarray,
    marker_names: list[str],
):
    """Plot conditional probability heatmap: P(B missing | A missing).

    Args:
        partial_data: Array of shape (N_partial, n_markers, 3) with frames
            where at least one marker is missing (NaN).
        marker_names: Labels for each marker.

    Returns:
        Tuple of (Figure, co-occurrence matrix).
    """
    n_markers = len(marker_names)
    missing_mask = np.isnan(partial_data[:, :, 0])

    co_occur = np.zeros((n_markers, n_markers))
    for a in range(n_markers):
        a_missing = missing_mask[:, a]
        n_a = a_missing.sum()
        if n_a == 0:
            continue
        for b in range(n_markers):
            co_occur[a, b] = (a_missing & missing_mask[:, b]).sum() / n_a

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(co_occur, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(n_markers))
    ax.set_yticks(range(n_markers))
    ax.set_xticklabels(marker_names, rotation=45, ha='right')
    ax.set_yticklabels(marker_names)
    ax.set_xlabel('Marker B')
    ax.set_ylabel('Marker A (missing)')
    ax.set_title('P(B missing | A missing)')

    for a in range(n_markers):
        for b in range(n_markers):
            colour = 'white' if co_occur[a, b] > 0.5 else 'black'
            ax.text(b, a, f'{co_occur[a, b]:.2f}', ha='center', va='center',
                    fontsize=10, color=colour)

    plt.colorbar(im, ax=ax, label='Conditional probability')
    fig.tight_layout()
    return fig, co_occur


def plot_ordering_violations(
    all_data: np.ndarray,
    violation_mask: np.ndarray,
    marker_a_idx: int,
    marker_b_idx: int,
    marker_a_name: str,
    marker_b_name: str,
):
    """Plot spatial distribution of ordering violations.

    Shows where in coordinate space the anatomical ordering is violated,
    overlaying violation frames (red) on the density of all frames (grey).

    Args:
        all_data: Array of shape (N, n_markers, 3) with all frames.
        violation_mask: Boolean array of shape (N,) marking violation frames.
        marker_a_idx: Index of first marker (e.g. wingtip = 0).
        marker_b_idx: Index of second marker (e.g. primary = 1).
        marker_a_name: Display name for first marker.
        marker_b_name: Display name for second marker.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for col, (coord_a, coord_b, lab_a, lab_b) in enumerate([
        (0, 2, 'x (lateral)', 'z (vertical)'),
        (0, 1, 'x (lateral)', 'y (longitudinal)'),
    ]):
        for row, (m, name) in enumerate([(marker_a_idx, marker_a_name), (marker_b_idx, marker_b_name)]):
            ax = axes[row, col]

            present = ~np.isnan(all_data[:, m, 0])
            a_all = all_data[present, m, coord_a]
            b_all = all_data[present, m, coord_b]

            ax.hist2d(a_all, b_all, bins=60, cmap='Greys', alpha=0.5, density=True)

            viol_present = violation_mask & present
            a_viol = all_data[viol_present, m, coord_a]
            b_viol = all_data[viol_present, m, coord_b]

            ax.scatter(a_viol, b_viol, c='red', s=0.1, alpha=0.2, rasterized=True)
            ax.set_xlabel(lab_a)
            ax.set_ylabel(lab_b)
            ax.set_title(f'{name} positions ({marker_b_name} ≥ {marker_a_name} in x)')
            ax.set_aspect('equal')

    fig.suptitle(f'Where do {marker_b_name}–{marker_a_name} ordering violations occur?', y=1.02)
    fig.tight_layout()
    return fig
