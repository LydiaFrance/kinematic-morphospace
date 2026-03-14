"""Tests for kinematic_morphospace.preprocessing.polygon_labelling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kinematic_morphospace.preprocessing.polygon_labelling import (
    _extract_bird_prefix,
    label_by_polygons,
)


@pytest.fixture
def square_boundaries():
    """Simple square boundaries for testing."""
    # Square from (-0.1, -0.1) to (0.1, 0.1) in both XY and YZ
    square = np.array([
        [-0.1, -0.1],
        [0.1, -0.1],
        [0.1, 0.1],
        [-0.1, 0.1],
    ])
    return {
        "TestBird": {
            "backpack": {"XY": square, "YZ": square},
        }
    }


@pytest.fixture
def sample_df():
    """Sample unlabelled marker DataFrame."""
    return pd.DataFrame({
        "seqID": ["TestBird_01"] * 5,
        "frameID": [f"TestBird_01_{i:06d}" for i in range(5)],
        "label": [""] * 5,
        "xyz_1": [0.0, 0.0, 0.5, 0.0, 0.0],   # X
        "xyz_2": [0.0, 0.0, 0.0, 0.5, 0.0],   # Y
        "xyz_3": [0.0, 0.0, 0.0, 0.0, 0.5],   # Z
    })


class TestLabelByPolygons:
    """Tests for label_by_polygons."""

    def test_inside_both_planes(self, square_boundaries, sample_df):
        result = label_by_polygons(
            sample_df, square_boundaries,
            bird_col="seqID",
        )
        # Point (0,0,0) is inside both XY and YZ squares
        assert result.loc[0, "label"] == "backpack"

    def test_outside_xy(self, square_boundaries, sample_df):
        result = label_by_polygons(
            sample_df, square_boundaries,
            bird_col="seqID",
        )
        # Point (0.5, 0, 0) is outside XY square
        assert result.loc[2, "label"] == ""

    def test_outside_yz(self, square_boundaries, sample_df):
        result = label_by_polygons(
            sample_df, square_boundaries,
            bird_col="seqID",
        )
        # Point (0, 0, 0.5) is outside YZ square
        assert result.loc[4, "label"] == ""

    def test_no_mutation_of_input(self, square_boundaries, sample_df):
        original = sample_df.copy()
        label_by_polygons(sample_df, square_boundaries, bird_col="seqID")
        pd.testing.assert_frame_equal(sample_df, original)

    def test_already_labelled_not_overwritten(self, square_boundaries):
        df = pd.DataFrame({
            "seqID": ["TestBird_01"],
            "frameID": ["TestBird_01_000000"],
            "label": ["tailpack"],
            "xyz_1": [0.0],
            "xyz_2": [0.0],
            "xyz_3": [0.0],
        })
        result = label_by_polygons(
            df, square_boundaries,
            bird_col="seqID",
        )
        assert result.loc[0, "label"] == "tailpack"

    def test_empty_dataframe(self, square_boundaries):
        df = pd.DataFrame(columns=["seqID", "frameID", "label", "xyz_1", "xyz_2", "xyz_3"])
        result = label_by_polygons(df, square_boundaries, bird_col="seqID")
        assert len(result) == 0

    def test_bird_id_map(self):
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1.0]])
        boundaries = {"Drogon": {"backpack": {"XY": square, "YZ": square}}}

        df = pd.DataFrame({
            "seqID": ["01_09_001"],
            "frameID": ["01_09_001_000000"],
            "label": [""],
            "xyz_1": [0.0],
            "xyz_2": [0.0],
            "xyz_3": [0.0],
        })
        result = label_by_polygons(
            df, boundaries,
            bird_col="seqID",
            bird_id_map={"01": "Drogon"},
        )
        assert result.loc[0, "label"] == "backpack"


class TestLateralise:
    """Tests for the lateralise parameter of label_by_polygons."""

    @pytest.fixture
    def wide_boundaries(self):
        """Boundaries covering x in [-1, 1]."""
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1.0]])
        return {"TestBird": {"wingtip": {"XY": square, "YZ": square}}}

    def test_left_prefix_when_x_negative(self, wide_boundaries):
        df = pd.DataFrame({
            "seqID": ["TestBird_01"],
            "label": [""],
            "xyz_1": [-0.5],
            "xyz_2": [0.0],
            "xyz_3": [0.0],
        })
        result = label_by_polygons(
            df, wide_boundaries, bird_col="seqID", lateralise=True
        )
        assert result.loc[0, "label"] == "left_wingtip"

    def test_right_prefix_when_x_positive(self, wide_boundaries):
        df = pd.DataFrame({
            "seqID": ["TestBird_01"],
            "label": [""],
            "xyz_1": [0.5],
            "xyz_2": [0.0],
            "xyz_3": [0.0],
        })
        result = label_by_polygons(
            df, wide_boundaries, bird_col="seqID", lateralise=True
        )
        assert result.loc[0, "label"] == "right_wingtip"

    def test_right_prefix_when_x_zero(self, wide_boundaries):
        df = pd.DataFrame({
            "seqID": ["TestBird_01"],
            "label": [""],
            "xyz_1": [0.0],
            "xyz_2": [0.0],
            "xyz_3": [0.0],
        })
        result = label_by_polygons(
            df, wide_boundaries, bird_col="seqID", lateralise=True
        )
        assert result.loc[0, "label"] == "right_wingtip"

    def test_lateralise_false_unchanged(self, wide_boundaries):
        df = pd.DataFrame({
            "seqID": ["TestBird_01"],
            "label": [""],
            "xyz_1": [-0.5],
            "xyz_2": [0.0],
            "xyz_3": [0.0],
        })
        result = label_by_polygons(
            df, wide_boundaries, bird_col="seqID", lateralise=False
        )
        assert result.loc[0, "label"] == "wingtip"

    def test_mixed_left_right(self, wide_boundaries):
        df = pd.DataFrame({
            "seqID": ["TestBird_01"] * 3,
            "label": [""] * 3,
            "xyz_1": [-0.5, 0.5, -0.3],
            "xyz_2": [0.0, 0.0, 0.0],
            "xyz_3": [0.0, 0.0, 0.0],
        })
        result = label_by_polygons(
            df, wide_boundaries, bird_col="seqID", lateralise=True
        )
        assert result.loc[0, "label"] == "left_wingtip"
        assert result.loc[1, "label"] == "right_wingtip"
        assert result.loc[2, "label"] == "left_wingtip"


class TestExtractBirdPrefix:

    def test_normal(self):
        assert _extract_bird_prefix("01_09_001") == "01"

    def test_short_string(self):
        assert _extract_bird_prefix("X") == "X"
