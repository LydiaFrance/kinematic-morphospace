"""Schematic diagrams for robustness validation notebook."""

from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba

# Default marker colour scheme
# MARKER_COLOURS = {
#     "wingtip":   "#de6b48",
#     "primary":   "#CEC075",
#     "secondary": "#7dbbc3",
#     "tailtip":   "#8f6593",
# }
# MARKER_COLOURS = {
#     "wingtip":   "#EA526F",
#     "primary":   "#FF8A5B",
#     "secondary": "#CFE795",
#     "tailtip":   "#25CED1",
# }
MARKER_COLOURS = {
    "wingtip":   "#C94059",
    "primary":   "#E07043",
    "secondary": "#B2CB76",
    "tailtip":   "#33999A",
}

# ── Shared grid primitives ──────────────────────────────────────────────

def _make_grid(n_frames, n_markers, n_axes, colours, alphas):
    """Build an original grid: grid[frame][marker] = [(colour, alpha)] * n_axes."""
    return [
        [[(colours[m], alphas[f])] * n_axes for m in range(n_markers)]
        for f in range(n_frames)
    ]


def _draw_grid_panel(ax, grid, marker_names, n_frames, n_markers, n_axes,
                     title, subtitle, show_marker_names, show_xyz,
                     show_frame_labels, strip_w, strip_gap, marker_gap,
                     cell_h, panel_w, held_out_col=None, aspect="equal"):
    """Draw one panel of the grid schematic on the given axes."""
    ax.set_xlim(-0.6, panel_w + 0.1)
    ax.set_ylim(-1.1, n_frames * cell_h + 1.0)
    ax.set_aspect(aspect)
    ax.axis("off")

    marker_w = n_axes * strip_w + (n_axes - 1) * strip_gap

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.text(panel_w / 2, -0.65, subtitle,
            ha="center", va="top", fontsize=8, color="0.4", linespacing=1.4)

    for m in range(n_markers):
        mx = m * (marker_w + marker_gap)
        if show_marker_names:
            ax.text(mx + marker_w / 2, n_frames * cell_h + 0.55,
                    marker_names[m], ha="center", va="bottom",
                    fontsize=7.5, color="0.35")
        if show_xyz:
            for a, label in enumerate(["x", "y", "z"]):
                sx = mx + a * (strip_w + strip_gap) + strip_w / 2
                ax.text(sx, n_frames * cell_h + 0.12, label,
                        ha="center", va="bottom", fontsize=6.5, color="0.5")

    if show_frame_labels:
        for f in range(n_frames):
            y = (n_frames - 1 - f) * cell_h
            ax.text(-0.15, y + cell_h / 2, f"F{f+1}",
                    ha="right", va="center", fontsize=7.5, color="0.4")

    for f in range(n_frames):
        for m in range(n_markers):
            mx = m * (marker_w + marker_gap)
            y = (n_frames - 1 - f) * cell_h
            is_held_out = (held_out_col is not None and m == held_out_col)

            for a in range(n_axes):
                colour_hex, alpha = grid[f][m][a]
                if is_held_out:
                    rgba = to_rgba("#cccccc", alpha=0.3)
                else:
                    rgba = to_rgba(colour_hex, alpha=alpha)
                sx = mx + a * (strip_w + strip_gap)
                rect = mpatches.FancyBboxPatch(
                    (sx + 0.02, y + 0.03), strip_w - 0.04, cell_h - 0.06,
                    boxstyle="round,pad=0.02", facecolor=rgba,
                    edgecolor="white", linewidth=0.8,
                )
                ax.add_patch(rect)


# ── Shuffle schematic ───────────────────────────────────────────────────

def _apply_shuffles(original, n_frames, n_markers, n_axes, colours, alphas,
                    seed=42):
    """Apply the four shuffle modes to the original grid."""
    rng = np.random.default_rng(seed)

    # Temporal: shuffle each marker's 3D vector across frames
    temporal = _make_grid(n_frames, n_markers, n_axes, colours, alphas)
    for m in range(n_markers):
        perm = rng.permutation(n_frames)
        for f in range(n_frames):
            temporal[f][m] = original[perm[f]][m][:]

    # Column: shuffle each scalar column independently
    column = _make_grid(n_frames, n_markers, n_axes, colours, alphas)
    for m in range(n_markers):
        for ax in range(n_axes):
            perm = rng.permutation(n_frames)
            for f in range(n_frames):
                column[f][m][ax] = original[perm[f]][m][ax]

    # Label: permute marker identities within each frame
    label = []
    for f in range(n_frames):
        perm = rng.permutation(n_markers)
        label.append([original[f][perm[m]][:] for m in range(n_markers)])

    # Complete: shuffle every scalar value in the matrix
    complete = _make_grid(n_frames, n_markers, n_axes, colours, alphas)
    all_values = [(colours[m], alphas[f])
                  for f in range(n_frames)
                  for m in range(n_markers)
                  for _ in range(n_axes)]
    perm = rng.permutation(len(all_values))
    shuffled = [all_values[p] for p in perm]
    idx = 0
    for f in range(n_frames):
        for m in range(n_markers):
            for a in range(n_axes):
                complete[f][m][a] = shuffled[idx]
                idx += 1

    return temporal, column, label, complete


def _layout_shuffle_schematic(axes, marker_colours=None, n_frames=5, seed=42,
                               aspect="equal"):
    """Draw the five-panel shuffle schematic onto provided axes.

    Args:
        axes: Array of 5 pre-created Axes to draw into (original + 4 shuffle modes).
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames: Number of toy frames to show. Defaults to 5.
        seed: RNG seed for reproducible shuffles. Defaults to 42.
        aspect: Axes aspect ratio string. Defaults to 'equal'.
    """
    mc = marker_colours or MARKER_COLOURS
    marker_names = list(mc.keys())
    colours = list(mc.values())
    n_markers = len(marker_names)
    n_axes = 3

    alphas = np.linspace(0.35, 1.0, n_frames)
    original = _make_grid(n_frames, n_markers, n_axes, colours, alphas)
    temporal, column, label, complete = _apply_shuffles(
        original, n_frames, n_markers, n_axes, colours, alphas, seed)

    panels = [
        ("Original",         original,  True,  True,  True),
        ("Temporal shuffle",  temporal,  True,  True,  False),
        ("Column shuffle",    column,    True,  True,  False),
        ("Label shuffle",     label,     False, True,  True),
        ("Complete shuffle",  complete,  False, False, False),
    ]
    subtitles = [
        "Rows = frames in time (light \u2192 dark)\nColours = marker identity",
        "Each marker shuffled in time\nBreaks between-marker coordination",
        "Each column shuffled independently\nBreaks within-marker spatial coherence",
        "Marker labels shuffled per frame\nBreaks anatomical identity",
        "Everything shuffled\nBreaks all structure",
    ]

    strip_w, strip_gap, marker_gap, cell_h = 0.28, 0.02, 0.25, 0.55
    marker_w = n_axes * strip_w + (n_axes - 1) * strip_gap
    panel_w = n_markers * marker_w + (n_markers - 1) * marker_gap

    for ax_idx, (ax, (title, grid, show_names, show_xyz, show_frames)) in \
            enumerate(zip(axes, panels, strict=False)):
        _draw_grid_panel(ax, grid, marker_names, n_frames, n_markers, n_axes,
                         title, subtitles[ax_idx], show_names, show_xyz,
                         show_frames, strip_w, strip_gap, marker_gap,
                         cell_h, panel_w, aspect=aspect)


def plot_shuffle_schematic(marker_colours=None, n_frames=5, seed=42):
    """Plot five-panel shuffle schematic showing each shuffle mode.

    Panels show the original data alongside temporal, column, label, and
    complete shuffle variants. Used to visually explain the four
    null-model perturbations applied in the robustness analysis.

    Args:
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames: Number of toy frames to show in each panel. Defaults to 5.
        seed: RNG seed for reproducible shuffle ordering. Defaults to 42.

    Returns:
        Figure containing the five-panel shuffle schematic.
    """
    cell_h = 0.55
    fig, axes = plt.subplots(1, 5, figsize=(20, n_frames * cell_h + 3.8),
                              gridspec_kw={"wspace": 0.32})
    _layout_shuffle_schematic(axes, marker_colours, n_frames, seed)

    # plt.suptitle("How each shuffle mode rearranges the data", fontsize=13,
    #              fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ── Subsampling schematic ──────────────────────────────────────────────

def _layout_subsampling_schematic(axes, marker_colours=None, n_frames=5,
                                   aspect="equal"):
    """Draw the five-panel marker subsampling schematic onto provided axes.

    Args:
        axes: Array of 5 pre-created Axes (full grid + one per held-out marker).
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames: Number of toy frames to show. Defaults to 5.
        aspect: Axes aspect ratio string. Defaults to 'equal'.
    """
    mc = marker_colours or MARKER_COLOURS
    marker_names = list(mc.keys())
    colours = list(mc.values())
    n_markers = len(marker_names)
    n_axes = 3

    alphas = np.linspace(0.35, 1.0, n_frames)
    original = _make_grid(n_frames, n_markers, n_axes, colours, alphas)

    panels = [
        ("All markers", None),
    ] + [
        (f"Drop {marker_names[m]}", m) for m in range(n_markers)
    ]
    subtitles = [
        "Full 4-marker basis",
    ] + [
        ""
        for _ in range(n_markers)
    ]

    strip_w, strip_gap, marker_gap, cell_h = 0.28, 0.02, 0.25, 0.55
    marker_w = n_axes * strip_w + (n_axes - 1) * strip_gap
    panel_w = n_markers * marker_w + (n_markers - 1) * marker_gap

    for ax_idx, (ax, (_title, held_out)) in enumerate(zip(axes, panels, strict=False)):
        _draw_grid_panel(ax, original, marker_names, n_frames, n_markers,
                         n_axes, "", subtitles[ax_idx],
                         show_marker_names=True, show_xyz=True,
                         show_frame_labels=(ax_idx == 0),
                         strip_w=strip_w, strip_gap=strip_gap,
                         marker_gap=marker_gap, cell_h=cell_h,
                         panel_w=panel_w, held_out_col=held_out,
                         aspect=aspect)


def plot_subsampling_schematic(marker_colours=None, n_frames=5):
    """Plot the five-panel marker subsampling schematic.

    Shows the original full-marker grid alongside four leave-one-out panels,
    each with one marker column greyed out to indicate it has been withheld.

    Args:
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames: Number of toy frames to show in each panel. Defaults to 5.

    Returns:
        Figure containing the leave-one-out subsampling schematic.
    """
    mc = marker_colours or MARKER_COLOURS
    n_markers = len(mc)
    n_panels = 1 + n_markers
    cell_h = 0.55
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(4 * n_panels, n_frames * cell_h + 0.5),
                              gridspec_kw={"wspace": 0.1})
    _layout_subsampling_schematic(axes, marker_colours, n_frames)

    plt.suptitle("Leave-one-out marker subsampling", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ── Relabelling schematic ──────────────────────────────────────────────

def _layout_relabelling_schematic(axes, marker_colours=None, n_frames=20,
                                   fractions=(0.05, 0.25), seed=42,
                                   aspect="equal"):
    """Draw the relabelling schematic onto provided axes.

    Args:
        axes: Array of 1 + len(fractions) pre-created Axes.
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames: Number of toy frames to show. Defaults to 20.
        fractions: Relabelling fractions to illustrate. Defaults to (0.05, 0.25).
        seed: RNG seed for reproducible frame selection. Defaults to 42.
        aspect: Axes aspect ratio string. Defaults to 'equal'.
    """
    mc = marker_colours or MARKER_COLOURS
    marker_names = list(mc.keys())
    colours = list(mc.values())
    n_markers = len(marker_names)
    n_axes = 3

    alphas = np.linspace(0.35, 1.0, n_frames)
    original = _make_grid(n_frames, n_markers, n_axes, colours, alphas)

    rng = np.random.default_rng(seed)

    relabelled_grids = {}
    swapped_masks = {}
    for frac in fractions:
        grid = _make_grid(n_frames, n_markers, n_axes, colours, alphas)
        n_swap = max(1, round(frac * n_frames))
        swap_frames = set(rng.choice(n_frames, size=n_swap, replace=False))
        swapped_masks[frac] = swap_frames

        for f in swap_frames:
            perm = rng.permutation(n_markers)
            grid[f] = [original[f][perm[m]][:] for m in range(n_markers)]

        relabelled_grids[frac] = grid

    strip_w, strip_gap, marker_gap, cell_h = 0.28, 0.02, 0.25, 0.35
    marker_w = n_axes * strip_w + (n_axes - 1) * strip_gap
    panel_w = n_markers * marker_w + (n_markers - 1) * marker_gap

    panels = [("Original", original, None)] + [
        (f"{frac:.0%} relabelled", relabelled_grids[frac], swapped_masks[frac])
        for frac in fractions
    ]
    subtitles = ["All frames correctly labelled"] + [
        f"{round(frac * n_frames)} of {n_frames} frames\nhave marker labels shuffled"
        for frac in fractions
    ]

    for ax_idx, (ax, (title, grid, swapped)) in enumerate(
        zip(axes, panels, strict=False)
    ):
        _draw_grid_panel(ax, grid, marker_names, n_frames, n_markers, n_axes,
                         title, subtitles[ax_idx],
                         show_marker_names=(ax_idx == 0),
                         show_xyz=(ax_idx == 0),
                         show_frame_labels=False,
                         strip_w=strip_w, strip_gap=strip_gap,
                         marker_gap=marker_gap, cell_h=cell_h,
                         panel_w=panel_w, aspect=aspect)

        if swapped:
            for f in list(swapped):
                y = (n_frames - 1 - f) * cell_h
                ax.plot(-0.25, y + cell_h / 2, 's', color='#EE7674',
                        markersize=3, alpha=0.7)


def plot_relabelling_schematic(marker_colours=None, n_frames=20,
                               fractions=(0.05, 0.25), seed=42):
    """Plot a schematic illustrating the effect of randomly swapping marker labels.

    Shows the original correctly-labelled data alongside panels where a given
    fraction of frames have had their marker labels permuted. Swapped frames are
    marked with a red square indicator on the left.

    Args:
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames: Number of toy frames to show (more frames = clearer proportion
            visual). Defaults to 20.
        fractions: Relabelling fractions to illustrate. Defaults to (0.05, 0.25).
        seed: RNG seed for reproducible frame selection. Defaults to 42.

    Returns:
        Figure containing the relabelling schematic.
    """
    n_panels = 1 + len(fractions)
    cell_h = 0.35
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(4 * n_panels, n_frames * cell_h + 0.5),
                              gridspec_kw={"wspace": 0.35})
    _layout_relabelling_schematic(axes, marker_colours, n_frames, fractions, seed)
    plt.tight_layout()
    return fig


# ── Missing data imputation schematic ──────────────────────────────────

def _layout_imputation_schematic(axes, marker_colours=None,
                                  n_frames_complete=10, n_frames_missing=20,
                                  missing_rates=(0.25, 0.33, 0.25, 0.21),
                                  seed=42, aspect="equal"):
    """Draw the three-panel missing-data imputation schematic onto provided axes.

    Args:
        axes: Array of 3 pre-created Axes (complete, missing, imputed).
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames_complete: Number of frames in the complete-data panel. Defaults to 10.
        n_frames_missing: Number of frames in the missing and imputed panels.
            Defaults to 20.
        missing_rates: Per-marker dropout rate matching the real data pattern.
            Defaults to (0.25, 0.33, 0.25, 0.21).
        seed: RNG seed for reproducible dropout masks. Defaults to 42.
        aspect: Axes aspect ratio string. Defaults to 'equal'.
    """
    mc = marker_colours or MARKER_COLOURS
    marker_names = list(mc.keys())
    colours = list(mc.values())
    n_markers = len(marker_names)
    n_axes = 3

    rng = np.random.default_rng(seed)

    strip_w, strip_gap, marker_gap, cell_h = 0.28, 0.02, 0.25, 0.35
    marker_w = n_axes * strip_w + (n_axes - 1) * strip_gap
    panel_w = n_markers * marker_w + (n_markers - 1) * marker_gap

    missing_mask = np.zeros((n_frames_missing, n_markers), dtype=bool)
    for m in range(n_markers):
        rate = missing_rates[m] if m < len(missing_rates) else 0.25
        n_drop = round(rate * n_frames_missing)
        drop_frames = rng.choice(n_frames_missing, size=n_drop, replace=False)
        missing_mask[drop_frames, m] = True

    alphas_complete = np.linspace(0.35, 1.0, n_frames_complete)
    alphas_missing = np.linspace(0.35, 1.0, n_frames_missing)

    grid_complete = _make_grid(n_frames_complete, n_markers, n_axes,
                                colours, alphas_complete)
    grid_missing = _make_grid(n_frames_missing, n_markers, n_axes,
                               colours, alphas_missing)
    grid_imputed = _make_grid(n_frames_missing, n_markers, n_axes,
                               colours, alphas_missing)

    panels = [
        ("Complete data", grid_complete, n_frames_complete, None, None),
        ("Missing data", grid_missing, n_frames_missing, missing_mask, "missing"),
        ("Imputed data", grid_imputed, n_frames_missing, missing_mask, "imputed"),
    ]
    subtitles = [
        "252,630 straight-flight frames\nAll markers present",
        "669,212 straight-flight frames (2.6×)\n29–46% dropout per marker",
        "Missing values estimated\nvia iterative imputation",
    ]

    for ax_idx, (ax, (_title, grid, n_f, mask, mode)) in enumerate(
        zip(axes, panels, strict=False)
    ):
        ax.set_xlim(-0.6, panel_w + 0.1)
        ax.set_ylim(-1.1, n_f * cell_h + 1.0)
        ax.set_aspect(aspect)
        ax.axis("off")

        ax.text(panel_w / 2, -0.65, subtitles[ax_idx],
                ha="center", va="top", fontsize=14, color="0.4",
                linespacing=1.4)

        if ax_idx == 0:
            for m in range(n_markers):
                mx = m * (marker_w + marker_gap)
                ax.text(mx + marker_w / 2, n_f * cell_h + 0.55,
                        marker_names[m], ha="center", va="bottom",
                        fontsize=10, color="0.35")
                for a, label in enumerate(["x", "y", "z"]):
                    sx = mx + a * (strip_w + strip_gap) + strip_w / 2
                    ax.text(sx, n_f * cell_h + 0.12, label,
                            ha="center", va="bottom", fontsize=8,
                            color="0.5")

        for f in range(n_f):
            for m in range(n_markers):
                mx = m * (marker_w + marker_gap)
                y = (n_f - 1 - f) * cell_h
                is_missing = (mask is not None and mask[f, m])

                for a in range(n_axes):
                    sx = mx + a * (strip_w + strip_gap)

                    if mode == "missing" and is_missing:
                        rect = mpatches.FancyBboxPatch(
                            (sx + 0.02, y + 0.03),
                            strip_w - 0.04, cell_h - 0.06,
                            boxstyle="round,pad=0.02",
                            facecolor="white",
                            edgecolor="0.8", linewidth=0.5,
                            linestyle="-",
                        )
                        ax.add_patch(rect)
                    elif mode == "imputed" and is_missing:
                        colour_hex, alpha = grid[f][m][a]
                        rgba = to_rgba(colour_hex, alpha=alpha * 0.6)
                        rect = mpatches.FancyBboxPatch(
                            (sx + 0.02, y + 0.03),
                            strip_w - 0.04, cell_h - 0.06,
                            boxstyle="round,pad=0.02",
                            facecolor=rgba,
                            edgecolor="0.5", linewidth=0.7,
                            linestyle="--",
                        )
                        ax.add_patch(rect)
                    else:
                        colour_hex, alpha = grid[f][m][a]
                        rgba = to_rgba(colour_hex, alpha=alpha)
                        rect = mpatches.FancyBboxPatch(
                            (sx + 0.02, y + 0.03),
                            strip_w - 0.04, cell_h - 0.06,
                            boxstyle="round,pad=0.02",
                            facecolor=rgba,
                            edgecolor="white", linewidth=0.8,
                        )
                        ax.add_patch(rect)


def plot_imputation_schematic(marker_colours=None, n_frames_complete=10,
                               n_frames_missing=20,
                               missing_rates=(0.25, 0.33, 0.25, 0.21),
                               seed=42):
    """Plot three-panel schematic of complete, missing, and imputed data.

    Each panel is a grid of colour-coded cells representing marker
    coordinates. Missing cells are shown as outlines; imputed cells are
    shown in a faded version of the original colour with a dashed border.

    Args:
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames_complete: Number of frames in the complete-data panel. Defaults to 10.
        n_frames_missing: Number of frames in the missing and imputed panels.
            Defaults to 20.
        missing_rates: Per-marker dropout rate matching the real data pattern.
            Defaults to (0.25, 0.33, 0.25, 0.21).
        seed: RNG seed for reproducible dropout masks. Defaults to 42.

    Returns:
        Figure containing the three-panel imputation schematic.
    """
    cell_h = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(12, max(n_frames_complete,
                              n_frames_missing) * cell_h + 1),
                              gridspec_kw={"wspace": 0.35})
    _layout_imputation_schematic(axes, marker_colours, n_frames_complete,
                                  n_frames_missing, missing_rates, seed)
    plt.tight_layout()
    return fig


# ── Pairwise distance schematic ────────────────────────────────────────

def _make_gradient(rgb_a, rgb_b, alpha, n_steps=32, vertical=False):
    """Build an RGBA image array for a linear colour gradient between two colours.

    Args:
        rgb_a: Start colour (hex string or RGB tuple).
        rgb_b: End colour (hex string or RGB tuple).
        alpha: Opacity applied uniformly across the gradient.
        n_steps: Number of interpolation steps. Defaults to 32.
        vertical: When True the gradient runs top-to-bottom (n_steps, 1, 4);
            when False it runs left-to-right (1, n_steps, 4). Defaults to False.

    Returns:
        RGBA array suitable for ax.imshow().
    """
    a = np.array(to_rgb(rgb_a))
    b = np.array(to_rgb(rgb_b))
    t = np.linspace(0, 1, n_steps)
    rgb = a[None, :] * (1 - t[:, None]) + b[None, :] * t[:, None]
    if vertical:
        rgba = np.ones((n_steps, 1, 4))
        rgba[:, 0, :3] = rgb
        rgba[:, 0, 3] = alpha
    else:
        rgba = np.ones((1, n_steps, 4))
        rgba[0, :, :3] = rgb
        rgba[0, :, 3] = alpha
    return rgba


def _draw_distance_panel(ax, grid, col_labels, n_frames, n_cols,
                         title, subtitle, show_col_labels, show_frame_labels,
                         col_w, col_gap, cell_h, panel_w,
                         sort_arrow=False, aspect="equal"):
    """Draw one pairwise-distance panel onto the given axes.

    Each cell is a vertical colour gradient from rgb_a (top) to rgb_b (bottom),
    representing a pairwise distance between two markers. The gradient encodes
    which two marker types are being compared.

    Args:
        ax: Axes to draw into.
        grid: Nested list grid[frame][col] = (rgb_a, rgb_b, alpha).
        col_labels: Column header strings, or None to suppress labels.
        n_frames: Number of data rows (frames).
        n_cols: Number of distance columns.
        title: Panel title displayed above the grid.
        subtitle: Descriptive text displayed below the grid.
        show_col_labels: Whether to show column header labels.
        show_frame_labels: Whether to show row labels (F1, F2, …).
        col_w: Width of each cell in data units.
        col_gap: Gap between cells in data units.
        cell_h: Height of each cell in data units.
        panel_w: Total panel width in data units.
        sort_arrow: When True, draws a directional annotation indicating
            that columns are ordered from greatest to smallest distance.
            Defaults to False.
        aspect: Axes aspect ratio string. Defaults to 'equal'.
    """
    ax.set_xlim(-0.6, panel_w + 0.1)
    ax.set_ylim(-1.1, n_frames * cell_h + 1.0)
    ax.set_aspect(aspect)
    ax.axis("off")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.text(panel_w / 2, -0.65, subtitle,
            ha="center", va="top", fontsize=8, color="0.4", linespacing=1.4)

    for c in range(n_cols):
        cx = c * (col_w + col_gap)
        if show_col_labels and col_labels is not None:
            ax.text(cx + col_w / 2, n_frames * cell_h + 0.15,
                    col_labels[c], ha="center", va="bottom",
                    fontsize=6, color="0.35", rotation=45)

    if sort_arrow:
        # Arrow spanning the full width of the columns
        arrow_y = n_frames * cell_h + 0.25
        x_start = 0.0
        x_end = panel_w
        ax.annotate("", xy=(x_start, arrow_y), xytext=(x_end, arrow_y),
                     arrowprops={"arrowstyle": "->", "color": "0.4", "lw": 0.7})
        ax.text(x_start, arrow_y + 0.12, "greatest",
                ha="left", va="bottom", fontsize=6.5, color="0.4")
        ax.text(x_end, arrow_y + 0.12, "smallest",
                ha="right", va="bottom", fontsize=6.5, color="0.4")

    if show_frame_labels:
        for f in range(n_frames):
            y = (n_frames - 1 - f) * cell_h
            ax.text(-0.15, y + cell_h / 2, f"F{f+1}",
                    ha="right", va="center", fontsize=7.5, color="0.4")

    for f in range(n_frames):
        for c in range(n_cols):
            cx = c * (col_w + col_gap)
            y = (n_frames - 1 - f) * cell_h
            rgb_a, rgb_b, alpha = grid[f][c]
            grad = _make_gradient(rgb_a, rgb_b, alpha, vertical=True)
            ax.imshow(grad, extent=[cx + 0.02, cx + col_w - 0.02,
                                    y + 0.03, y + cell_h - 0.03],
                      aspect="auto", interpolation="bilinear", zorder=1)
            # White border to match the other panels
            rect = mpatches.FancyBboxPatch(
                (cx + 0.02, y + 0.03), col_w - 0.04, cell_h - 0.06,
                boxstyle="round,pad=0.02", facecolor="none",
                edgecolor="white", linewidth=0.8,
            )
            ax.add_patch(rect)


def _layout_pairwise_distance_schematic(fig, gridspec_region,
                                         marker_colours=None, n_frames=5,
                                         seed=42, aspect="equal"):
    """Draw the pairwise-distance schematic into a GridSpec region.

    Renders four column groups: marker coordinates, labelled pairwise distances,
    sorted pairwise distances, and within-frame-shuffled pairwise distances.
    An explainer inset illustrates how the six C(4,2) pairwise distances are
    derived from the four marker positions.

    Args:
        fig: Parent Figure to add axes to.
        gridspec_region: SubplotSpec region to draw into (via subgridspec).
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames: Number of toy frames to display in each panel. Defaults to 5.
        seed: RNG seed for reproducible shuffle ordering. Defaults to 42.
        aspect: Axes aspect ratio string. Defaults to 'equal'.
    """
    mc = marker_colours or MARKER_COLOURS
    marker_names = list(mc.keys())
    colours = list(mc.values())
    n_markers = len(marker_names)
    n_axes = 3

    alphas = np.linspace(0.35, 1.0, n_frames)

    strip_w, strip_gap, marker_gap_xyz, cell_h = 0.28, 0.02, 0.25, 0.55
    marker_w = n_axes * strip_w + (n_axes - 1) * strip_gap
    panel_w_xyz = n_markers * marker_w + (n_markers - 1) * marker_gap_xyz
    original_grid = _make_grid(n_frames, n_markers, n_axes, colours, alphas)

    pairs = list(combinations(range(n_markers), 2))
    n_pairs = len(pairs)
    pair_labels = [f"{marker_names[a][:3]}-{marker_names[b][:3]}"
                   for a, b in pairs]
    pair_endpoints = [(colours[a], colours[b]) for a, b in pairs]

    pw_grid = [[(pair_endpoints[p][0], pair_endpoints[p][1], alphas[f])
                for p in range(n_pairs)]
               for f in range(n_frames)]

    rng = np.random.default_rng(seed)
    sorted_grid = []
    for f in range(n_frames):
        order = rng.permutation(n_pairs)
        sorted_grid.append([pw_grid[f][order[p]] for p in range(n_pairs)])

    shuffled_grid = []
    for f in range(n_frames):
        perm = rng.permutation(n_pairs)
        shuffled_grid.append([pw_grid[f][perm[p]] for p in range(n_pairs)])

    col_w = 0.4
    col_gap = 0.05
    panel_w_dist = n_pairs * (col_w + col_gap) - col_gap

    dist_panels = [
        (
            "Labelled\npairwise distances",
            pw_grid,
            pair_labels,
            True,
            True,
            False,
        ),
        (
            "Sorted\npairwise distances",
            sorted_grid,
            None,
            False,
            True,
            True,
        ),
        (
            "Shuffled\npairwise distances",
            shuffled_grid,
            None,
            False,
            True,
            False,
        ),
    ]
    dist_subtitles = [
        "Ordered with labels",
        "Ordered by size per frame",
        "Order shuffled per frame",
    ]

    gs = gridspec_region.subgridspec(2, 4, width_ratios=[1.3, 1, 1, 1],
                                      height_ratios=[3, 1.5],
                                      wspace=0.3, hspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    _draw_grid_panel(ax0, original_grid, marker_names, n_frames, n_markers,
                     n_axes, "Marker coordinates", "",
                     show_marker_names=True, show_xyz=True,
                     show_frame_labels=True,
                     strip_w=strip_w, strip_gap=strip_gap,
                     marker_gap=marker_gap_xyz, cell_h=cell_h,
                     panel_w=panel_w_xyz, aspect=aspect)

    ax_exp = fig.add_subplot(gs[1, 0])
    ax_exp.set_xlim(-0.3, max(panel_w_xyz, panel_w_dist) + 0.3)
    ax_exp.set_ylim(-2.0, 1.8)
    ax_exp.set_aspect(aspect)
    ax_exp.axis("off")

    triplet_y = 0.8
    triplet_h = cell_h * 0.8
    marker_centres = []
    demo_alpha = 0.85
    for m in range(n_markers):
        mx = m * (marker_w + marker_gap_xyz)
        marker_centres.append(mx + marker_w / 2)
        for a_idx in range(n_axes):
            sx = mx + a_idx * (strip_w + strip_gap)
            rgba = to_rgba(colours[m], alpha=demo_alpha)
            rect = mpatches.FancyBboxPatch(
                (sx + 0.02, triplet_y), strip_w - 0.04, triplet_h,
                boxstyle="round,pad=0.02", facecolor=rgba,
                edgecolor="white", linewidth=0.8,
            )
            ax_exp.add_patch(rect)
        ax_exp.text(mx + marker_w / 2, triplet_y + triplet_h + 0.1,
                    marker_names[m], ha="center", va="bottom",
                    fontsize=6.5, color="0.35")

    dist_cell_w = col_w * 0.8
    dist_total_w = n_pairs * dist_cell_w + (n_pairs - 1) * col_gap * 0.8
    dist_x_start = (panel_w_xyz - dist_total_w) / 2
    dist_cell_y = -1.2
    dist_cell_h = cell_h * 0.7

    for p, (a, b) in enumerate(pairs):
        cx = dist_x_start + p * (dist_cell_w + col_gap * 0.8)
        grad = _make_gradient(colours[a], colours[b], demo_alpha, vertical=True)
        ax_exp.imshow(grad, extent=[cx, cx + dist_cell_w,
                                     dist_cell_y, dist_cell_y + dist_cell_h],
                      aspect="auto", interpolation="bilinear", zorder=1)
        rect = mpatches.FancyBboxPatch(
            (cx, dist_cell_y), dist_cell_w, dist_cell_h,
            boxstyle="round,pad=0.02", facecolor="none",
            edgecolor="white", linewidth=0.8,
        )
        ax_exp.add_patch(rect)

        cell_mid_x = cx + dist_cell_w / 2
        cell_top_y = dist_cell_y + dist_cell_h
        for m_idx, colour in [(a, colours[a]), (b, colours[b])]:
            ax_exp.annotate(
                "", xy=(cell_mid_x, cell_top_y),
                xytext=(marker_centres[m_idx], triplet_y),
                arrowprops={"arrowstyle": "-", "color": colour,
                                "lw": 0.6, "alpha": 0.5,
                                "connectionstyle": "arc3,rad=0.15"},
            )

    ax_exp.text(panel_w_xyz / 2, dist_cell_y - 0.15,
                "C(4,2) = 6 pairwise distances",
                ha="center", va="top", fontsize=7.5, color="0.4")

    for i, (col, (title, grid, labels, show_labels, show_frames, arrow)) in \
            enumerate(zip([1, 2, 3], dist_panels, strict=False)):
        ax = fig.add_subplot(gs[:, col])
        _draw_distance_panel(ax, grid, labels, n_frames, n_pairs,
                             title, dist_subtitles[i], show_labels,
                             show_frames, col_w, col_gap, cell_h,
                             panel_w_dist, sort_arrow=arrow)


def plot_pairwise_distance_schematic(marker_colours=None, n_frames=5, seed=42):
    """Plot the pairwise-distance schematic comparing three distance representations.

    Shows marker coordinates alongside labelled, sorted, and within-frame-shuffled
    pairwise distance representations. Used to illustrate the three pairwise-distance
    variants tested in the robustness analysis.

    Args:
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.
        n_frames: Number of toy frames to show in each panel. Defaults to 5.
        seed: RNG seed for reproducible distance shuffles. Defaults to 42.

    Returns:
        Figure containing the pairwise-distance schematic.
    """
    cell_h = 0.55
    fig = plt.figure(figsize=(14, n_frames * cell_h + 2))
    gs = fig.add_gridspec(1, 1)
    _layout_pairwise_distance_schematic(fig, gs[0, 0], marker_colours,
                                         n_frames, seed)
    return fig



# ── Autocorrelation schematic ─────────────────────────────────────────

def _draw_sequence_dots(ax, x_centre, n_frames, marker_colours,
                        seq_colour, seq_label, dot_size=8,
                        row_spacing=0.18, marker_spacing=0.35,
                        greyed=None, max_y=None):
    """Draw one sequence as a vertical column of marker dots.

    Each row is one frame; each column is one marker (coloured dot).
    Greyed frames are drawn as light grey dots.
    """
    greyed = greyed or set()
    n_markers = len(marker_colours)
    total_w = (n_markers - 1) * marker_spacing
    total_h = (n_frames - 1) * row_spacing

    # y_top is passed in by the caller so all sequences align at the top.
    # Dots are drawn top-down: frame 0 at y_top, last frame at y_top - total_h.
    y_top = max_y if max_y is not None else total_h
    y_bottom = y_top - total_h

    # Background band
    pad_x, pad_y = 0.15, 0.08
    bg = mpatches.FancyBboxPatch(
        (x_centre - total_w / 2 - pad_x, y_bottom - pad_y),
        total_w + 2 * pad_x, total_h + 2 * pad_y,
        boxstyle="round,pad=0.06",
        facecolor=to_rgba(seq_colour, alpha=0.2),
        edgecolor=to_rgba(seq_colour, alpha=0.5),
        linewidth=1.0,
    )
    ax.add_patch(bg)

    # Sequence label underneath
    ax.text(x_centre, y_bottom - pad_y - 0.2, seq_label,
            fontsize=7.5, fontweight="bold", color=to_rgb(seq_colour),
            ha="center", va="top")

    # Draw dots (top-down: frame 0 at top)
    ms = dot_size ** 0.5
    for ff in range(n_frames):
        yy = y_top - ff * row_spacing
        for mm, col in enumerate(marker_colours):
            xx = x_centre + (mm - (n_markers - 1) / 2) * marker_spacing
            if ff in greyed:
                ax.plot(xx, yy, "o", color="#cccccc", markersize=ms,
                        markeredgecolor="white", markeredgewidth=0.3,
                        zorder=2)
            else:
                ax.plot(xx, yy, "o", color=col, markersize=ms,
                        markeredgecolor="white", markeredgewidth=0.3,
                        zorder=3)

    return total_h


def _layout_autocorrelation_schematic(axes, marker_colours=None):
    """Draw two-panel autocorrelation schematic.

    Panel 1: Original — sequences shown as columns of small marker dots.
    Panel 2: Thinned — same sequences, every 3rd frame kept, others grey.
    """
    mc = marker_colours or MARKER_COLOURS
    colours = list(mc.values())
    marker_names = list(mc.keys())

    seq_defs: list[dict[str, int | str]] = [
        {"n_frames": 18, "colour": "#8FBBD9", "label": "Seq 1"},
        {"n_frames": 12, "colour": "#E8927C", "label": "Seq 2"},
        {"n_frames": 20, "colour": "#8EC89A", "label": "Seq 3"},
        {"n_frames": 8,  "colour": "#C9A9D4", "label": "Seq 4"},
        {"n_frames": 15, "colour": "#E8C87A", "label": "Seq 5"},
    ]

    dot_size = 10
    row_spacing = 0.18
    marker_spacing = 0.30
    seq_spacing = 2.0
    n_frames_list = [int(sd["n_frames"]) for sd in seq_defs]
    max_frames = max(n_frames_list)
    total_h = (max_frames - 1) * row_spacing
    n_seqs = len(seq_defs)

    def _draw_panel(ax, title, subtitle, thin_step=None):
        for ss, sd in enumerate(seq_defs):
            x_centre = ss * seq_spacing
            n_frames_val = sd["n_frames"]
            assert isinstance(n_frames_val, int)
            greyed = ({ff for ff in range(n_frames_val)
                       if ff % thin_step != 0} if thin_step else None)
            _draw_sequence_dots(
                ax, x_centre, sd["n_frames"], colours,
                sd["colour"], sd["label"],
                dot_size=dot_size, row_spacing=row_spacing,
                marker_spacing=marker_spacing, greyed=greyed,
                max_y=total_h,
            )

        total_w = (n_seqs - 1) * seq_spacing
        ax.set_xlim(-1.5, total_w + 1.5)
        ax.set_ylim(-1.2, total_h + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.text(total_w / 2, -1.0, subtitle,
                ha="center", va="top", fontsize=8, color="0.4")

    # Panel 1: Original
    _draw_panel(axes[0], "Original", "All frames")

    # Marker legend at top
    for mm, (name, col) in enumerate(zip(marker_names, colours, strict=False)):
        lx = mm * 1.8
        axes[0].plot(lx, total_h + 0.35, "o", color=col,
                     markersize=dot_size ** 0.5 * 0.8,
                     markeredgecolor="white", markeredgewidth=0.3)
        axes[0].text(lx + 0.2, total_h + 0.35, name, fontsize=6.5,
                     color="0.4", va="center")

    # Panel 2: Every 2nd frame
    _draw_panel(axes[1], "Every 2nd frame", "50% of frames",
                thin_step=2)

    # Panel 3: Every 20th frame
    _draw_panel(axes[2], "Every 20th frame", "5% of frames",
                thin_step=20)


def plot_autocorrelation_schematic(marker_colours=None):
    """Plot a three-panel schematic showing temporal thinning of flight sequences.

    Illustrates the autocorrelation robustness analysis by showing how sequences
    look when using all frames, every 2nd frame (50%), and every 20th frame (5%).
    Each sequence is shown as a vertical column of coloured marker dots.

    Args:
        marker_colours: Dict of {name: hex} for each marker. Defaults to
            MARKER_COLOURS.

    Returns:
        Figure containing the three-panel thinning schematic.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5),
        gridspec_kw={"wspace": 0.12},
    )
    _layout_autocorrelation_schematic(axes, marker_colours)
    plt.tight_layout()
    return fig
