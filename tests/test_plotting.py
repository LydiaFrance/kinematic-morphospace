"""Smoke tests for plotting helpers used in manuscript figures (Q17).

These tests verify that the plotting functions run without error on
synthetic data and return the expected types.  They do NOT assert
visual correctness — that is validated by notebook execution.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Suppress the non-interactive backend warning from plt.show() calls
# inside the plotting functions under test.
pytestmark = [
    pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic data matching the schemas expected by the
# plotting modules.
# ---------------------------------------------------------------------------

PC_COLS = [f"PC{i:02}" for i in range(1, 13)]


def _make_scores_df(n_frames=200, n_flights=10, seed=0):
    """Build a minimal scores DataFrame matching the pipeline schema."""
    rng = np.random.default_rng(seed)
    n_per_flight = n_frames // n_flights

    rows = []
    for seq_i in range(n_flights):
        bird_id = (seq_i % 5) + 1
        year = 2017 if seq_i < n_flights // 2 else 2020
        obstacle = 1 if seq_i % 3 == 0 else 0
        for frame_j in range(n_per_flight):
            fid = f"{bird_id:02}_{seq_i:03}_{frame_j:06}"
            sid = f"{bird_id:02}_{seq_i:03}"
            horz = -8.0 + frame_j * (8.0 / n_per_flight)
            row = {
                "frameID": fid,
                "seqID": sid,
                "BirdID": bird_id,
                "Year": year,
                "Obstacle": obstacle,
                "Left": frame_j % 2 == 0,
                "HorzDistance": horz,
                "PerchDistance": 9,
                "Naive": 0,
                "Turn": "Straight",
                "time": frame_j * 0.01,
                "VertDistance": rng.normal(0, 0.1),
                "body_pitch": rng.normal(0, 5),
                "IMU": 0,
            }
            for pc in PC_COLS:
                row[pc] = rng.normal(0, 0.1)
            rows.append(row)

    return pd.DataFrame(rows)


def _make_binned_scores_df(scores_df):
    """Add a 'bins' column mimicking bin_by_horz_distance output."""
    df = scores_df.copy()
    df["bins"] = pd.cut(df["HorzDistance"], bins=20, labels=False).astype(float)
    return df


@pytest.fixture
def scores_df():
    return _make_scores_df()


@pytest.fixture
def binned_scores_df(scores_df):
    return _make_binned_scores_df(scores_df)


@pytest.fixture
def score_limits():
    score_5 = pd.Series({pc: -0.3 for pc in PC_COLS})
    score_95 = pd.Series({pc: 0.3 for pc in PC_COLS})
    score_mid = pd.Series({pc: 0.0 for pc in PC_COLS})
    return score_5, score_95, score_mid


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ===========================================================================
# variance.py
# ===========================================================================

class TestVariancePlotting:
    """Smoke tests for kinematic_morphospace.plotting.variance."""

    def test_plot_explained_returns_axes(self):
        from kinematic_morphospace.plotting.variance import plot_explained
        ratios = np.array([0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.01])
        fig, ax = plot_explained(ratios, annotate=False, colour_before=len(ratios))
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_explained_with_annotation(self):
        from kinematic_morphospace.plotting.variance import plot_explained
        ratios = np.array([0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.01])
        fig, ax = plot_explained(ratios, annotate=True, colour_before=len(ratios))
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_explained_comparison(self):
        from kinematic_morphospace.plotting.variance import plot_explained_comparison
        real = np.array([0.4, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.01])
        shuffled = np.full(9, 1.0 / 9)
        fig, ax = plot_explained_comparison(real, shuffled)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_cumulative_variance_ratios(self):
        from kinematic_morphospace.plotting.variance import plot_cumulative_variance_ratios
        data = {
            "Drogon_2017_straight": np.cumsum(np.array([0.4, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01])),
            "Ruby_2020_obs": np.cumsum(np.array([0.35, 0.2, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01, 0.01])),
        }
        fig = plot_cumulative_variance_ratios(data)
        assert isinstance(fig, Figure)

    def test_plot_hist_similar_shapes(self):
        from kinematic_morphospace.plotting.variance import plot_hist_similar_shapes
        rng = np.random.default_rng(42)
        n_frames, n_markers, n_components = 50, 4, 4
        marker_data = rng.standard_normal((n_frames, n_markers, 3))
        flat = marker_data.reshape(n_frames, -1)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components).fit(flat)
        scores = pca.transform(flat)
        fig, axes = plot_hist_similar_shapes(
            pca.components_, scores, marker_data,
            pc_indices=[0, 1], threshold=0.5,
        )
        assert isinstance(fig, Figure)


# ===========================================================================
# symmetry.py
# ===========================================================================

class TestSymmetryPlotting:
    """Smoke tests for kinematic_morphospace.plotting.symmetry."""

    def test_major_axis_regression_returns_tuple(self):
        from kinematic_morphospace.plotting.symmetry import _major_axis_regression
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 2))
        slope, intercept, pct_var = _major_axis_regression(data)
        assert isinstance(slope, float)
        assert isinstance(intercept, float)
        assert 0 <= pct_var <= 100

    def test_plot_left_right(self):
        from kinematic_morphospace.plotting.symmetry import plot_left_right
        rng = np.random.default_rng(0)
        # Build left_right_scores with required columns
        n = 100
        data = {}
        for pc in PC_COLS:
            data[f"{pc}_left"] = rng.normal(0, 0.1, n)
            data[f"{pc}_right"] = rng.normal(0, 0.1, n)
        lr_df = pd.DataFrame(data)

        score_5 = pd.Series({pc: -0.3 for pc in PC_COLS})
        score_95 = pd.Series({pc: 0.3 for pc in PC_COLS})

        # Should complete without error
        plot_left_right(lr_df, score_5, score_95)

    def test_plot_left_right_just_two(self):
        from kinematic_morphospace.plotting.symmetry import plot_left_right_just_two
        rng = np.random.default_rng(0)
        n = 100
        data = {}
        for pc in PC_COLS:
            data[f"{pc}_left"] = rng.normal(0, 0.1, n)
            data[f"{pc}_right"] = rng.normal(0, 0.1, n)
        lr_df = pd.DataFrame(data)

        score_5 = pd.Series({pc: -0.3 for pc in PC_COLS})
        score_95 = pd.Series({pc: 0.3 for pc in PC_COLS})

        plot_left_right_just_two(lr_df, score_5, score_95)

    def test_plot_left_right_empty(self):
        from kinematic_morphospace.plotting.symmetry import plot_left_right_empty
        score_5 = pd.Series({pc: -0.3 for pc in PC_COLS})
        score_95 = pd.Series({pc: 0.3 for pc in PC_COLS})
        fig, ax = plot_left_right_empty(score_5, score_95, PC=0)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_plot_symmetry_scores(self):
        from kinematic_morphospace.plotting.symmetry import plot_symmetry_scores
        scores = np.array([0.01, 0.02, 0.03, 0.04, 0.06, 0.08,
                           0.10, 0.12, 0.15, 0.18, 0.20, 0.22])
        plot_symmetry_scores(scores, threshold=0.05)


# ===========================================================================
# heatmaps.py
# ===========================================================================

class TestHeatmapPlotting:
    """Smoke tests for kinematic_morphospace.plotting.heatmaps."""

    def test_plot_PC_score_heatmaps(self, score_limits):
        from kinematic_morphospace.plotting.heatmaps import plot_PC_score_heatmaps
        score_5, score_95, score_mid = score_limits
        rng = np.random.default_rng(0)

        # Build a binned scores DataFrame
        n = 200
        data = {"bins": pd.cut(np.linspace(-8, -0.5, n), bins=20, labels=False).astype(float)}
        for pc in PC_COLS:
            data[pc] = rng.normal(0, 0.1, n)
        df = pd.DataFrame(data)

        ax = plot_PC_score_heatmaps(df, PC_COLS[:4], score_5, score_95, score_mid, "Test")
        assert isinstance(ax, Axes)

    def test_plot_difference_PC_scores_heatmap(self, score_limits):
        from kinematic_morphospace.plotting.heatmaps import plot_difference_PC_scores_heatmap
        score_5, score_95, _ = score_limits
        rng = np.random.default_rng(0)

        n = 200
        bins = pd.cut(np.linspace(-8, -0.5, n), bins=20, labels=False).astype(float)
        data_ctrl = {"bins": bins}
        data_exp = {"bins": bins}
        for pc in PC_COLS:
            data_ctrl[pc] = rng.normal(0, 0.1, n)
            data_exp[pc] = rng.normal(0.05, 0.1, n)
        df_ctrl = pd.DataFrame(data_ctrl)
        df_exp = pd.DataFrame(data_exp)

        ax = plot_difference_PC_scores_heatmap(
            df_ctrl, df_exp, PC_COLS[:4], score_5, score_95)
        assert isinstance(ax, Axes)

    def test_plot_difference_exp_scores_heatmap(self, score_limits):
        from kinematic_morphospace.plotting.heatmaps import plot_difference_exp_scores_heatmap
        score_5, score_95, _ = score_limits
        rng = np.random.default_rng(0)

        n = 200
        bins = pd.cut(np.linspace(-8, -0.5, n), bins=20, labels=False).astype(float)
        data_ctrl = {"bins": bins}
        data_exp = {"bins": bins}
        for pc in PC_COLS:
            data_ctrl[pc] = rng.normal(0, 0.1, n)
            data_exp[pc] = rng.normal(0.05, 0.1, n)
        df_ctrl = pd.DataFrame(data_ctrl)
        df_exp = pd.DataFrame(data_exp)

        ax = plot_difference_exp_scores_heatmap(
            df_ctrl, "Control", df_exp, "Experiment",
            PC_COLS[:4], score_5, score_95)
        assert isinstance(ax, Axes)


# ===========================================================================
# save_figure (plotting/__init__.py)
# ===========================================================================

class TestSaveFigure:
    """Tests for save_figure including rasterize support."""

    def test_save_figure_pdf(self, tmp_path):
        from kinematic_morphospace.plotting import save_figure
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = tmp_path / "test.pdf"
        save_figure(fig, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_figure_rasterize(self, tmp_path):
        from kinematic_morphospace.plotting import save_figure
        fig, ax = plt.subplots()
        ax.scatter(range(100), range(100))
        path = tmp_path / "test_raster.pdf"
        save_figure(fig, path, rasterize=True)
        assert path.exists()
        assert path.stat().st_size > 0
        # Verify rasterize was restored (no side-effect)
        assert not ax.get_rasterized()

    def test_save_figure_accepts_tuple(self, tmp_path):
        from kinematic_morphospace.plotting import save_figure
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = tmp_path / "test_tuple.pdf"
        save_figure((fig, ax), path)
        assert path.exists()
