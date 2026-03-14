"""
Tests for the kinematic_morphospace.preprocessing subpackage.

Uses small synthetic DataFrames to test each transformation function
independently. Does NOT require actual .mat files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kinematic_morphospace.preprocessing.calibration import (
    apply_time_offsets,
    calibrate_position,
    calibrate_time,
    find_jump_frame,
)
from kinematic_morphospace.preprocessing.harmonise import (
    add_metadata,
    extract_bird_id,
    extract_seq_id,
    harmonise_labelled,
    harmonise_trajectory,
    join_body_pitch,
    join_smooth_xyz,
)
from kinematic_morphospace.preprocessing.mat_loader import (
    load_intermediate_csvs,
    matlab_table_to_dataframe,
)
from kinematic_morphospace.preprocessing.pipeline import PreprocessingConfig, run_from_csvs
from kinematic_morphospace.preprocessing.shape_tables import (
    create_bilateral_table,
    create_unilateral_table,
    filter_pure_side_frames,
    mirror_left_markers,
    pivot_markers_wide,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def sample_frame_ids():
    """Sample frameID strings from both campaigns."""
    return pd.Series([
        "D1_9m_seq1_001",
        "D3_9m_seq1_002",
        "D5_9m_seq2_003",
        "D4_5m_seq1_042",
    ])


@pytest.fixture()
def sample_trajectory_2020():
    """Minimal 2020 trajectory DataFrame."""
    return pd.DataFrame({
        "frameID": ["D3_9m_seq1_001", "D3_9m_seq1_002", "D5_9m_seq2_003"],
        "seqID": ["D3_9m_seq1", "D3_9m_seq1", "D5_9m_seq2"],
        "time": [0.1, 0.2, 0.15],
        "HorzDistance": [8.29, 8.31, 8.30],
        "XYZ_1": [1.0, 2.0, 3.0],
        "XYZ_2": [4.0, 5.0, 6.0],
        "XYZ_3": [2.5, 3.0, 2.8],
        "smooth_XYZ_1": [1.1, 2.1, 3.1],
        "smooth_XYZ_2": [4.1, 5.1, 6.1],
        "smooth_XYZ_3": [2.6, 3.1, 2.9],
    })


@pytest.fixture()
def sample_trajectory_2017():
    """Minimal 2017 trajectory DataFrame."""
    return pd.DataFrame({
        "frameID": ["D1_5m_seq1_001", "D3_5m_seq1_002"],
        "seqID": ["D1_5m_seq1", "D3_5m_seq1"],
        "BirdID": [1, 3],
        "PerchDistance": [5, 5],
        "time": [0.3, 0.4],
        "HorzDistance": [4.5, 4.8],
        "XYZ_1": [10.0, 11.0],
        "XYZ_2": [12.0, 13.0],
        "XYZ_3": [1.0, 1.5],
        "label": ["mean_backpack", "mean_backpack"],
        "frame": [1, 2],
    })


@pytest.fixture()
def sample_info_2020():
    """Minimal asymInfo table."""
    return pd.DataFrame({
        "seqID": ["D3_9m_seq1", "D5_9m_seq2"],
        "Obstacle": [1, 0],
        "IMU": [0, 1],
    })


@pytest.fixture()
def sample_tail():
    """Minimal tail/tailpack table."""
    return pd.DataFrame({
        "frameID": ["D3_9m_seq1_001", "D3_9m_seq1_002"],
        "body_pitch": [5.2, 5.5],
    })


@pytest.fixture()
def sample_smooth():
    """Minimal smooth body table for 2017."""
    return pd.DataFrame({
        "frameID": ["D1_5m_seq1_001", "D3_5m_seq1_002"],
        "XYZ_1": [10.1, 11.1],
        "XYZ_2": [12.1, 13.1],
        "XYZ_3": [1.1, 1.6],
    })


@pytest.fixture()
def sample_labelled_long():
    """Labelled marker data in long format with 4 marker types per side."""
    rows = []
    frame_id = "D3_9m_seq1_001"
    for side in ["left", "right"]:
        for marker in ["wingtip", "primary", "secondary", "tailtip"]:
            name = f"{side}_{marker}"
            x = 0.5 if side == "right" else -0.5
            rows.append({
                "frameID": frame_id,
                "seqID": "D3_9m_seq1",
                "MarkerName": name,
                "rot_xyz_1": x,
                "rot_xyz_2": 0.1,
                "rot_xyz_3": 0.2,
                "time": 0.1,
                "HorzDistance": 8.3,
                "body_pitch": 5.0,
                "BirdID": 3,
                "PerchDistance": 9,
                "Year": 2020,
                "Naive": 0,
                "Obstacle": 1,
                "IMU": 0,
                "backpack_smooth_XYZ_3": 2.5,
            })
    # Add a second frame with only right markers (incomplete)
    frame_id2 = "D3_9m_seq1_002"
    for marker in ["wingtip", "primary", "secondary", "tailtip"]:
        rows.append({
            "frameID": frame_id2,
            "seqID": "D3_9m_seq1",
            "MarkerName": f"right_{marker}",
            "rot_xyz_1": 0.6,
            "rot_xyz_2": 0.15,
            "rot_xyz_3": 0.25,
            "time": 0.2,
            "HorzDistance": 8.31,
            "body_pitch": 5.5,
            "BirdID": 3,
            "PerchDistance": 9,
            "Year": 2020,
            "Naive": 0,
            "Obstacle": 1,
            "IMU": 0,
            "backpack_smooth_XYZ_3": 2.6,
        })
    return pd.DataFrame(rows)


# ===================================================================
# mat_loader tests
# ===================================================================


class TestMatlabTableToDataframe:
    """Tests for converting MATLAB struct-like dicts to DataFrames."""

    def test_simple_dict(self):
        data = {"frameID": np.array(["f1", "f2"]), "time": np.array([0.1, 0.2])}
        df = matlab_table_to_dataframe(data)
        assert list(df.columns) == ["frameID", "time"]
        assert len(df) == 2

    def test_multicolumn_field(self):
        """An Nx3 array should be split into _1, _2, _3 columns."""
        data = {
            "frameID": np.array(["f1", "f2"]),
            "XYZ": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        }
        df = matlab_table_to_dataframe(data)
        assert "XYZ_1" in df.columns
        assert "XYZ_2" in df.columns
        assert "XYZ_3" in df.columns
        assert "XYZ" not in df.columns
        assert df["XYZ_1"].tolist() == [1.0, 4.0]

    def test_object_array_strings(self):
        """Object arrays (cell arrays) should become string columns."""
        data = {"names": np.array(["alpha", "beta"], dtype=object)}
        df = matlab_table_to_dataframe(data)
        assert df["names"].tolist() == ["alpha", "beta"]


# ===================================================================
# harmonise tests
# ===================================================================


class TestExtractBirdId:
    def test_basic(self, sample_frame_ids):
        result = extract_bird_id(sample_frame_ids)
        assert result.tolist() == [1, 3, 5, 4]

    def test_single(self):
        result = extract_bird_id(pd.Series(["D3_seq1_001"]))
        assert result.iloc[0] == 3


class TestExtractSeqId:
    def test_basic(self, sample_frame_ids):
        result = extract_seq_id(sample_frame_ids)
        expected = ["D1_9m_seq1", "D3_9m_seq1", "D5_9m_seq2", "D4_5m_seq1"]
        assert result.tolist() == expected

    def test_two_segments(self):
        result = extract_seq_id(pd.Series(["D3_001"]))
        assert result.iloc[0] == "D3"


class TestAddMetadata:
    def test_2020_metadata(self, sample_trajectory_2020, sample_info_2020):
        result = add_metadata(
            sample_trajectory_2020, year=2020, info_df=sample_info_2020
        )
        assert "BirdID" in result.columns
        assert "Year" in result.columns
        assert "Obstacle" in result.columns
        assert "IMU" in result.columns
        assert "Naive" in result.columns
        # BirdID extracted from 2nd char
        assert result["BirdID"].tolist() == [3, 3, 5]
        # Year
        assert (result["Year"] == 2020).all()
        # Naive: default 0 for 2020, except BirdID 5 (Charmander)
        assert result.loc[result["BirdID"] == 5, "Naive"].iloc[0] == 1
        assert result.loc[result["BirdID"] == 3, "Naive"].iloc[0] == 0

    def test_2017_metadata(self, sample_trajectory_2017):
        result = add_metadata(sample_trajectory_2017, year=2017)
        assert (result["Year"] == 2017).all()
        # Naive: default 1 for 2017, except BirdID 3 (Ruby)
        assert result.loc[result["BirdID"] == 3, "Naive"].iloc[0] == 0
        assert result.loc[result["BirdID"] == 1, "Naive"].iloc[0] == 1
        # Obstacle and IMU should be 0
        assert (result["Obstacle"] == 0).all()
        assert (result["IMU"] == 0).all()

    def test_obstacle_lookup(self, sample_trajectory_2020, sample_info_2020):
        result = add_metadata(
            sample_trajectory_2020, year=2020, info_df=sample_info_2020
        )
        # D3_9m_seq1 should have Obstacle=1
        seq1_mask = result["seqID"] == "D3_9m_seq1"
        assert (result.loc[seq1_mask, "Obstacle"] == 1).all()
        # D5_9m_seq2 should have Obstacle=0
        seq2_mask = result["seqID"] == "D5_9m_seq2"
        assert (result.loc[seq2_mask, "Obstacle"] == 0).all()

    def test_imu_lookup(self, sample_trajectory_2020, sample_info_2020):
        result = add_metadata(
            sample_trajectory_2020, year=2020, info_df=sample_info_2020
        )
        # D5_9m_seq2 should have IMU=1
        seq2_mask = result["seqID"] == "D5_9m_seq2"
        assert (result.loc[seq2_mask, "IMU"] == 1).all()

    def test_perch_distance_default_2020(self, sample_trajectory_2020):
        result = add_metadata(sample_trajectory_2020, year=2020)
        assert (result["PerchDistance"] == 9.0).all()


class TestJoinBodyPitch:
    def test_basic_join(self, sample_trajectory_2020, sample_tail):
        result = join_body_pitch(sample_trajectory_2020, sample_tail)
        assert "body_pitch" in result.columns
        # First two rows should have values, third should be NaN
        assert result["body_pitch"].iloc[0] == 5.2
        assert result["body_pitch"].iloc[1] == 5.5
        assert pd.isna(result["body_pitch"].iloc[2])

    def test_preserves_all_rows(self, sample_trajectory_2020, sample_tail):
        result = join_body_pitch(sample_trajectory_2020, sample_tail)
        assert len(result) == len(sample_trajectory_2020)


class TestJoinSmoothXyz:
    def test_basic_join(self, sample_trajectory_2017, sample_smooth):
        result = join_smooth_xyz(sample_trajectory_2017, sample_smooth)
        assert "smooth_XYZ_1" in result.columns
        assert "smooth_XYZ_2" in result.columns
        assert "smooth_XYZ_3" in result.columns
        assert len(result) == len(sample_trajectory_2017)


class TestHarmoniseTrajectory:
    def test_2020_harmonisation(
        self, sample_trajectory_2020, sample_info_2020, sample_tail
    ):
        result = harmonise_trajectory(
            sample_trajectory_2020,
            year=2020,
            info_df=sample_info_2020,
            tail_df=sample_tail,
        )
        # Should have all metadata columns
        for col in ["frameID", "seqID", "Year", "Obstacle", "IMU", "Naive",
                     "BirdID", "PerchDistance"]:
            assert col in result.columns, f"Missing column: {col}"
        # Mass should be NaN for 2020
        assert "mass" in result.columns
        assert result["mass"].isna().all()
        # Label should be set
        assert (result["label"] == "mean_backpack").all()

    def test_2017_harmonisation(
        self, sample_trajectory_2017, sample_tail, sample_smooth
    ):
        result = harmonise_trajectory(
            sample_trajectory_2017,
            year=2017,
            tail_df=sample_tail,
            smooth_df=sample_smooth,
        )
        assert "Year" in result.columns
        assert (result["Year"] == 2017).all()


class TestHarmoniseLabelled:
    def test_2017_adds_seq_id(self):
        df = pd.DataFrame({
            "frameID": ["D1_5m_seq1_001", "D3_5m_seq1_002"],
            "BirdID": [1, 3],
            "PerchDistance": [5, 5],
            "time": [0.1, 0.2],
        })
        result = harmonise_labelled(df, year=2017)
        assert "seqID" in result.columns
        assert result["seqID"].tolist() == ["D1_5m_seq1", "D3_5m_seq1"]


# ===================================================================
# calibration tests
# ===================================================================


class TestCalibratePosition:
    def test_subtracts_perch_height(self, sample_trajectory_2020):
        result = calibrate_position(sample_trajectory_2020, perch_height=1.25)
        np.testing.assert_allclose(
            result["XYZ_3"].values,
            sample_trajectory_2020["XYZ_3"].values - 1.25,
        )
        np.testing.assert_allclose(
            result["smooth_XYZ_3"].values,
            sample_trajectory_2020["smooth_XYZ_3"].values - 1.25,
        )

    def test_does_not_modify_original(self, sample_trajectory_2020):
        original_z = sample_trajectory_2020["XYZ_3"].copy()
        calibrate_position(sample_trajectory_2020, perch_height=1.25)
        pd.testing.assert_series_equal(
            sample_trajectory_2020["XYZ_3"], original_z
        )

    def test_custom_columns(self):
        df = pd.DataFrame({"my_z": [10.0, 20.0]})
        result = calibrate_position(df, perch_height=5.0, z_columns=["my_z"])
        assert result["my_z"].tolist() == [5.0, 15.0]


class TestFindJumpFrame:
    def test_exact_match(self):
        df = pd.DataFrame({
            "HorzDistance": [7.0, 8.0, 8.3, 9.0],
            "time": [0.1, 0.2, 0.3, 0.4],
        })
        result = find_jump_frame(df, jump_dist=8.3)
        assert result == 0.3

    def test_within_tight_tolerance(self):
        df = pd.DataFrame({
            "HorzDistance": [7.0, 8.0, 8.29, 9.0],
            "time": [0.1, 0.2, 0.3, 0.4],
        })
        result = find_jump_frame(df, jump_dist=8.3, tolerances=(0.02, 0.05))
        assert result == 0.3  # within 0.02 tolerance

    def test_wide_tolerance_uses_mean(self):
        df = pd.DataFrame({
            "HorzDistance": [7.0, 8.15, 8.25, 9.0],
            "time": [0.1, 0.2, 0.3, 0.4],
        })
        # Neither within 0.02 nor 0.05, but both within 0.2
        result = find_jump_frame(df, jump_dist=8.3, tolerances=(0.02, 0.05, 0.2))
        assert result == pytest.approx(0.25)  # mean(0.2, 0.3)

    def test_no_match_returns_nan(self):
        df = pd.DataFrame({
            "HorzDistance": [1.0, 2.0, 3.0],
            "time": [0.1, 0.2, 0.3],
        })
        result = find_jump_frame(df, jump_dist=8.3)
        assert np.isnan(result)


class TestCalibrateTime:
    def test_subtracts_per_sequence(self):
        df = pd.DataFrame({
            "seqID": ["s1", "s1", "s2", "s2"],
            "HorzDistance": [8.29, 8.31, 7.0, 8.30],
            "time": [1.0, 1.1, 2.0, 2.5],
        })
        result, offsets = calibrate_time(df, jump_dist=8.3)
        # Sequence s1: closest to 8.3 is 8.31 at t=1.1 (within tol 0.02)
        # Actually 8.31 is closest (diff=0.01), so t0=1.1
        assert result.loc[result["seqID"] == "s1", "time"].iloc[0] == pytest.approx(
            1.0 - 1.1
        )
        assert len(offsets) == 2

    def test_returns_offset_table(self):
        df = pd.DataFrame({
            "seqID": ["s1", "s1"],
            "HorzDistance": [8.3, 8.5],
            "time": [0.5, 0.6],
        })
        _, offsets = calibrate_time(df, jump_dist=8.3)
        assert "seqID" in offsets.columns
        assert "time_offset" in offsets.columns
        assert offsets["time_offset"].iloc[0] == 0.5


class TestApplyTimeOffsets:
    def test_applies_offsets(self):
        df = pd.DataFrame({
            "seqID": ["s1", "s1", "s2"],
            "time": [1.0, 2.0, 3.0],
        })
        offsets = pd.DataFrame({
            "seqID": ["s1", "s2"],
            "time_offset": [0.5, 1.0],
        })
        result = apply_time_offsets(df, offsets)
        assert result["time"].tolist() == [0.5, 1.5, 2.0]

    def test_skips_nan_offsets(self):
        df = pd.DataFrame({
            "seqID": ["s1", "s2"],
            "time": [1.0, 2.0],
        })
        offsets = pd.DataFrame({
            "seqID": ["s1", "s2"],
            "time_offset": [0.5, float("nan")],
        })
        result = apply_time_offsets(df, offsets)
        assert result["time"].iloc[0] == 0.5
        assert result["time"].iloc[1] == 2.0  # unchanged


# ===================================================================
# shape_tables tests
# ===================================================================


class TestMirrorLeftMarkers:
    def test_negates_left_x(self):
        df = pd.DataFrame({
            "MarkerName": ["left_wingtip", "right_wingtip", "left_primary"],
            "rot_xyz_1": [0.5, 0.5, 0.3],
            "rot_xyz_2": [1.0, 1.0, 1.0],
            "rot_xyz_3": [2.0, 2.0, 2.0],
        })
        result = mirror_left_markers(df, coord_prefix="rot_xyz")
        assert result["rot_xyz_1"].tolist() == [-0.5, 0.5, -0.3]
        # Y and Z unchanged
        assert result["rot_xyz_2"].tolist() == [1.0, 1.0, 1.0]
        assert result["rot_xyz_3"].tolist() == [2.0, 2.0, 2.0]


class TestFilterPureSideFrames:
    def test_keeps_pure_sides(self):
        df = pd.DataFrame({
            "frameID": ["f1", "f2", "f3"],
            "MarkerName_wingtip": [
                "left_wingtip", "right_wingtip", "left_wingtip",
            ],
            "MarkerName_primary": [
                "left_primary", "right_primary", "right_primary",
            ],
        })
        cols = ["MarkerName_wingtip", "MarkerName_primary"]
        result, is_left = filter_pure_side_frames(df, cols, n_markers=2)
        # f1 = all left, f2 = all right, f3 = mixed -> removed
        assert len(result) == 2
        assert is_left.iloc[0]
        assert not is_left.iloc[1]


class TestPivotMarkersWide:
    def test_basic_pivot(self):
        info = pd.DataFrame({
            "frameID": ["f1"],
            "time": [0.1],
        })
        markers = pd.DataFrame({
            "frameID": ["f1", "f1"],
            "MarkerName": ["right_wingtip", "right_primary"],
            "rot_xyz_1": [0.5, 0.3],
            "rot_xyz_2": [1.0, 1.1],
            "rot_xyz_3": [2.0, 2.1],
        })
        result, name_cols = pivot_markers_wide(
            info, markers, ["wingtip", "primary"],
            coord_prefix="rot_xyz", use_contains=True,
        )
        assert len(result) == 1
        assert "wingtip_rot_xyz_1" in result.columns
        assert "primary_rot_xyz_1" in result.columns
        assert len(name_cols) == 2

    def test_inner_join_drops_incomplete(self):
        info = pd.DataFrame({
            "frameID": ["f1", "f2"],
            "time": [0.1, 0.2],
        })
        markers = pd.DataFrame({
            "frameID": ["f1", "f1", "f2"],
            "MarkerName": ["right_wingtip", "right_primary", "right_wingtip"],
            "rot_xyz_1": [0.5, 0.3, 0.6],
            "rot_xyz_2": [1.0, 1.1, 1.2],
            "rot_xyz_3": [2.0, 2.1, 2.2],
        })
        result, _ = pivot_markers_wide(
            info, markers, ["wingtip", "primary"],
            coord_prefix="rot_xyz", use_contains=True,
        )
        # f2 has no primary marker, so it should be dropped by inner join
        assert len(result) == 1
        assert result["frameID"].iloc[0] == "f1"


class TestCreateUnilateralTable:
    def test_basic_creation(self, sample_labelled_long):
        result = create_unilateral_table(sample_labelled_long)
        # Should have wide-format columns
        assert "wingtip_rot_xyz_1" in result.columns
        assert "primary_rot_xyz_1" in result.columns
        assert "secondary_rot_xyz_1" in result.columns
        assert "tailtip_rot_xyz_1" in result.columns
        assert "Left" in result.columns
        # No MarkerName columns should remain
        marker_name_cols = [c for c in result.columns if c.startswith("MarkerName")]
        assert len(marker_name_cols) == 0

    def test_left_markers_mirrored(self, sample_labelled_long):
        result = create_unilateral_table(sample_labelled_long)
        # Left-side rows should have positive X (mirrored from -0.5 to 0.5)
        left_rows = result[result["Left"] == 1]
        if len(left_rows) > 0:
            assert left_rows["wingtip_rot_xyz_1"].iloc[0] > 0

    def test_has_vert_distance(self, sample_labelled_long):
        result = create_unilateral_table(sample_labelled_long)
        assert "VertDistance" in result.columns

    def test_pure_side_only(self, sample_labelled_long):
        result = create_unilateral_table(sample_labelled_long)
        # No mixed-side frames should exist
        assert "Left" in result.columns
        assert set(result["Left"].unique()).issubset({0, 1})


class TestCreateBilateralTable:
    def test_basic_creation(self, sample_labelled_long):
        result = create_bilateral_table(sample_labelled_long)
        # Should have both left and right marker columns
        assert "left_wingtip_rot_xyz_1" in result.columns
        assert "right_wingtip_rot_xyz_1" in result.columns
        # No Left column
        assert "Left" not in result.columns

    def test_no_mirroring(self, sample_labelled_long):
        result = create_bilateral_table(sample_labelled_long)
        # Left wingtip should retain original negative X
        if len(result) > 0:
            assert result["left_wingtip_rot_xyz_1"].iloc[0] == pytest.approx(-0.5)

    def test_only_complete_frames(self, sample_labelled_long):
        result = create_bilateral_table(sample_labelled_long)
        # Frame 2 only has right-side markers, so it should be dropped
        # (inner join requires all 8 markers)
        assert "D3_9m_seq1_002" not in result["frameID"].values


# ===================================================================
# pipeline config tests
# ===================================================================


class TestPreprocessingConfig:
    def test_defaults(self):
        config = PreprocessingConfig()
        assert config.perch_height == 1.25
        assert config.jump_dist == 8.3
        assert config.tolerances == (0.02, 0.05, 0.2)
        assert config.include_unrotated is False

    def test_custom_values(self):
        config = PreprocessingConfig(
            perch_height=1.5,
            jump_dist=9.0,
            data_dir_2020="/tmp/2020",
        )
        assert config.perch_height == 1.5
        assert config.jump_dist == 9.0
        assert str(config.data_dir_2020) == "/tmp/2020"


# ===================================================================
# CSV loading tests
# ===================================================================


def _make_traj_csv(n_rows: int = 3, year: int = 2020) -> pd.DataFrame:
    """Create a minimal trajectory DataFrame matching intermediate CSV schema."""
    bird = 3 if year == 2020 else 1
    return pd.DataFrame({
        "frameID": [f"D{bird}_9m_seq1_{i:03d}" for i in range(1, n_rows + 1)],
        "seqID": [f"D{bird}_9m_seq1"] * n_rows,
        "BirdID": [bird] * n_rows,
        "Year": [year] * n_rows,
        "time": [0.1 * i for i in range(1, n_rows + 1)],
        "HorzDistance": [8.3] * n_rows,
        "XYZ_1": np.random.default_rng(42).random(n_rows),
        "XYZ_2": np.random.default_rng(42).random(n_rows),
        "XYZ_3": np.random.default_rng(42).random(n_rows),
    })


def _make_labelled_csv(year: int = 2020) -> pd.DataFrame:
    """Create a minimal labelled DataFrame matching intermediate CSV schema."""
    bird = 3 if year == 2020 else 1
    frame_id = f"D{bird}_9m_seq1_001"
    rows = []
    for side in ["left", "right"]:
        for marker in ["wingtip", "primary", "secondary", "tailtip"]:
            x = 0.5 if side == "right" else -0.5
            rows.append({
                "frameID": frame_id,
                "seqID": f"D{bird}_9m_seq1",
                "MarkerName": f"{side}_{marker}",
                "rot_xyz_1": x,
                "rot_xyz_2": 0.1,
                "rot_xyz_3": 0.2,
                "time": 0.1,
                "HorzDistance": 8.3,
                "body_pitch": 5.0,
                "BirdID": bird,
                "PerchDistance": 9,
                "Year": year,
                "Naive": 0,
                "Obstacle": 0,
                "IMU": 0,
                "backpack_smooth_XYZ_3": 2.5,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def csv_dir(tmp_path):
    """Create a temp directory with synthetic intermediate CSV files."""
    prefix = "2024-03-24-"
    _make_traj_csv(3, year=2017).to_csv(
        tmp_path / f"{prefix}Traj2017.csv", index=False
    )
    _make_traj_csv(4, year=2020).to_csv(
        tmp_path / f"{prefix}Traj2020.csv", index=False
    )
    _make_labelled_csv(year=2017).to_csv(
        tmp_path / f"{prefix}Labelled2017.csv", index=False
    )
    _make_labelled_csv(year=2020).to_csv(
        tmp_path / f"{prefix}Labelled2020.csv", index=False
    )
    return tmp_path


class TestLoadIntermediateCsvs:
    def test_returns_four_keys(self, csv_dir):
        result = load_intermediate_csvs(csv_dir)
        assert set(result.keys()) == {
            "traj_2017", "traj_2020", "labelled_2017", "labelled_2020",
        }

    def test_dataframe_shapes(self, csv_dir):
        result = load_intermediate_csvs(csv_dir)
        assert len(result["traj_2017"]) == 3
        assert len(result["traj_2020"]) == 4
        assert len(result["labelled_2017"]) == 8  # 4 markers x 2 sides
        assert len(result["labelled_2020"]) == 8

    def test_custom_prefix(self, tmp_path):
        df = _make_traj_csv(2)
        df.to_csv(tmp_path / "myprefix-Traj2017.csv", index=False)
        df.to_csv(tmp_path / "myprefix-Traj2020.csv", index=False)
        df.to_csv(tmp_path / "myprefix-Labelled2017.csv", index=False)
        df.to_csv(tmp_path / "myprefix-Labelled2020.csv", index=False)
        result = load_intermediate_csvs(tmp_path, date_prefix="myprefix-")
        assert len(result) == 4

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_intermediate_csvs(tmp_path)


class TestRunFromCsvs:
    def test_output_keys(self, csv_dir):
        result = run_from_csvs(csv_dir)
        assert "trajectory" in result
        assert "labelled" in result
        assert "unilateral" in result
        assert "bilateral" in result

    def test_trajectory_combined(self, csv_dir):
        result = run_from_csvs(csv_dir)
        # 3 rows (2017) + 4 rows (2020) = 7
        assert len(result["trajectory"]) == 7

    def test_labelled_combined(self, csv_dir):
        result = run_from_csvs(csv_dir)
        # 8 rows (2017) + 8 rows (2020) = 16
        assert len(result["labelled"]) == 16

    def test_unilateral_has_expected_columns(self, csv_dir):
        result = run_from_csvs(csv_dir)
        uni = result["unilateral"]
        assert "wingtip_rot_xyz_1" in uni.columns
        assert "Left" in uni.columns

    def test_bilateral_has_expected_columns(self, csv_dir):
        result = run_from_csvs(csv_dir)
        bi = result["bilateral"]
        assert "left_wingtip_rot_xyz_1" in bi.columns
        assert "right_wingtip_rot_xyz_1" in bi.columns

    def test_include_unrotated(self, csv_dir):
        # Add xyz_* columns to labelled CSVs so unrotated path works
        prefix = "2024-03-24-"
        for name in ["Labelled2017", "Labelled2020"]:
            path = csv_dir / f"{prefix}{name}.csv"
            df = pd.read_csv(path)
            df["xyz_1"] = df["rot_xyz_1"]
            df["xyz_2"] = df["rot_xyz_2"]
            df["xyz_3"] = df["rot_xyz_3"]
            df.to_csv(path, index=False)

        result = run_from_csvs(csv_dir, include_unrotated=True)
        assert "unilateral_unrotated" in result
        assert "bilateral_unrotated" in result

    def test_saves_csvs(self, csv_dir, tmp_path):
        out_dir = tmp_path / "output"
        run_from_csvs(csv_dir, output_dir=out_dir)
        assert out_dir.exists()
        csv_files = list(out_dir.glob("*.csv"))
        assert len(csv_files) >= 4
