"""Tests for kinematic_morphospace.preprocessing.whole_body_pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kinematic_morphospace.preprocessing.marker_labelling import (
    filter_by_distance,
    fix_mislabelled_tailpack,
)
from kinematic_morphospace.preprocessing.coord_transform import compute_relative_positions
from kinematic_morphospace.preprocessing.smoothing import smooth_trajectory_with_gaps


class TestFixMislabelledTailpack:
    """Tests for fix_mislabelled_tailpack."""

    def test_relabels_ahead_tailpack(self):
        df = pd.DataFrame({
            "label": ["tailpack", "tailpack", "backpack"],
            "xyz_2": [0.1, -0.1, 0.0],  # first is ahead
        })
        result = fix_mislabelled_tailpack(df)
        assert result.loc[0, "label"] == "headpack"
        assert result.loc[1, "label"] == "tailpack"
        assert result.loc[2, "label"] == "backpack"

    def test_no_mutation(self):
        df = pd.DataFrame({
            "label": ["tailpack"],
            "xyz_2": [0.1],
        })
        original = df.copy()
        fix_mislabelled_tailpack(df)
        pd.testing.assert_frame_equal(df, original)


class TestFilterByDistance:
    """Tests for filter_by_distance."""

    def test_removes_too_far(self):
        df = pd.DataFrame({
            "label": ["backpack", "backpack", "backpack"],
            "xyz_1": [0.01, 0.03, 0.1],
            "xyz_2": [0.0, 0.0, 0.0],
            "xyz_3": [0.0, 0.0, 0.0],
        })
        result = filter_by_distance(df, "backpack", 0.0, 0.05)
        assert result.loc[0, "label"] == "backpack"
        assert result.loc[1, "label"] == "backpack"
        assert result.loc[2, "label"] == ""

    def test_removes_too_close(self):
        df = pd.DataFrame({
            "label": ["tailpack", "tailpack"],
            "xyz_1": [0.05, 0.2],
            "xyz_2": [0.0, 0.0],
            "xyz_3": [0.0, 0.0],
        })
        result = filter_by_distance(df, "tailpack", 0.1, 0.4)
        assert result.loc[0, "label"] == ""
        assert result.loc[1, "label"] == "tailpack"

    def test_no_matching_label(self):
        df = pd.DataFrame({
            "label": ["backpack"],
            "xyz_1": [0.0],
            "xyz_2": [0.0],
            "xyz_3": [0.0],
        })
        result = filter_by_distance(df, "tailpack", 0.1, 0.4)
        assert result.loc[0, "label"] == "backpack"


class TestComputeRelativePositions:
    """Tests for compute_relative_positions."""

    def test_subtraction(self):
        df = pd.DataFrame({
            "frameID": ["f1", "f2"],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
            "Z": [5.0, 6.0],
        })
        smooth = pd.DataFrame({
            "frameID": ["f1", "f2"],
            "smooth_X": [0.5, 1.0],
            "smooth_Y": [1.5, 2.0],
            "smooth_Z": [2.5, 3.0],
        })
        result = compute_relative_positions(df, smooth)
        np.testing.assert_allclose(result["xyz_1"].values, [0.5, 1.0])
        np.testing.assert_allclose(result["xyz_2"].values, [1.5, 2.0])
        np.testing.assert_allclose(result["xyz_3"].values, [2.5, 3.0])

    def test_inner_join(self):
        df = pd.DataFrame({
            "frameID": ["f1", "f2", "f3"],
            "X": [1.0, 2.0, 3.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 0.0],
        })
        smooth = pd.DataFrame({
            "frameID": ["f1", "f3"],
            "smooth_X": [0.5, 1.5],
            "smooth_Y": [0.0, 0.0],
            "smooth_Z": [0.0, 0.0],
        })
        result = compute_relative_positions(df, smooth)
        assert len(result) == 2
        assert set(result["frameID"]) == {"f1", "f3"}


class TestSmoothTrajectoryWithGaps:
    """Tests for smooth_trajectory_with_gaps."""

    def test_basic_smoothing(self):
        n = 100
        frames = np.arange(n)
        time = frames / 200.0
        xyz = np.column_stack([
            np.sin(time * 2 * np.pi),
            np.cos(time * 2 * np.pi),
            time,
        ])

        result = smooth_trajectory_with_gaps(
            time, frames, xyz,
            rms=0.001,
            frame_rate=200.0,
            max_gap_frames=30,
        )

        assert "smooth" in result
        assert "velocity" in result
        assert "acceleration" in result
        assert result["smooth"].shape[1] == 3

    def test_gap_detection(self):
        # Create data with a 50-frame gap
        frames = np.concatenate([np.arange(0, 50), np.arange(100, 150)])
        time = frames / 200.0
        xyz = np.column_stack([
            np.zeros(100),
            time,
            np.zeros(100),
        ])

        result = smooth_trajectory_with_gaps(
            time, frames, xyz,
            rms=0.001,
            frame_rate=200.0,
            max_gap_frames=30,
        )

        assert len(result["gaps"]) == 1
        # np.diff reports the gap as frame_diff (100 - 49 = 51)
        assert result["gaps"][0]["size"] > 30

    def test_output_keys(self):
        frames = np.arange(50)
        time = frames / 200.0
        xyz = np.zeros((50, 3))

        result = smooth_trajectory_with_gaps(time, frames, xyz)

        expected_keys = {"frames", "time", "smooth", "velocity", "acceleration", "gaps"}
        assert set(result.keys()) == expected_keys
