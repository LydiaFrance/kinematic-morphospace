"""
Tests for the C3D preprocessing modules (upstream pipeline).

Uses synthetic data — does NOT require actual C3D files or ezc3d.
Tests each module independently: stationary detection, trial splitting,
marker labelling, smoothing, coordinate transforms, time sync, body pitch,
and the C3DConfig dataclass.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# Fixtures — synthetic marker data
# ===================================================================


@pytest.fixture()
def synthetic_markers():
    """Synthetic marker table: 5 markers x 100 frames.

    Markers 0-2: stationary (perch markers, range < 0.001 m)
    Markers 3-4: moving (body markers, large range)
    """
    rng = np.random.default_rng(42)
    rows = []
    for frame in range(100):
        # Stationary markers (tiny noise)
        for mid in range(3):
            rows.append({
                "frame": frame,
                "marker_id": mid,
                "marker_label": f"marker_{mid}",
                "X": -1.0 + mid * 0.5 + rng.normal(0, 0.0001),
                "Y": -6.7 + mid * 4.5 + rng.normal(0, 0.0001),
                "Z": 1.25 + rng.normal(0, 0.0001),
                "residual": 1.0,
            })
        # Moving markers (body)
        for mid in [3, 4]:
            y_base = -6.0 + frame * 0.08  # moving along Y
            rows.append({
                "frame": frame,
                "marker_id": mid,
                "marker_label": f"marker_{mid}",
                "X": 0.1 * np.sin(frame / 10) + rng.normal(0, 0.01),
                "Y": y_base + rng.normal(0, 0.01),
                "Z": 1.5 + 0.1 * np.cos(frame / 10) + rng.normal(0, 0.01),
                "residual": 1.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def body_markers_with_distances():
    """Markers with known pairwise distances matching backpack bins.

    3 markers forming distances ~0.0165 and ~0.0335 m (within backpack bins).
    """
    rows = []
    for frame in range(20):
        # Marker A at origin
        rows.append({
            "frame": frame, "marker_id": 10,
            "X": 0.0, "Y": 0.0, "Z": 0.0,
        })
        # Marker B at 0.0165 m (within [0.016, 0.017])
        rows.append({
            "frame": frame, "marker_id": 11,
            "X": 0.0165, "Y": 0.0, "Z": 0.0,
        })
        # Marker C at 0.0335 m (within [0.033, 0.034])
        rows.append({
            "frame": frame, "marker_id": 12,
            "X": 0.0335, "Y": 0.0, "Z": 0.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def body_stats_df():
    """Synthetic body statistics for coord transforms and time sync."""
    frames = np.arange(200)
    # Bird starts at Y ≈ -8.5 (near 9m perch) and flies toward Y = 0
    y = -8.5 + frames * 0.04
    return pd.DataFrame({
        "frame": frames,
        "mean_X": np.zeros(200),
        "mean_Y": y,
        "mean_Z": np.ones(200) * 1.5,
        "smooth_X": np.zeros(200),
        "smooth_Y": y,
        "smooth_Z": np.ones(200) * 1.5,
        "vel_X": np.zeros(200),
        "vel_Y": np.ones(200) * 8.0,
        "vel_Z": np.zeros(200),
        "speed": np.ones(200) * 8.0,
    })


# ===================================================================
# stationary.py tests
# ===================================================================


class TestComputeMarkerMovement:
    def test_returns_one_row_per_marker(self, synthetic_markers):
        from kinematic_morphospace.preprocessing.stationary import compute_marker_movement

        result = compute_marker_movement(synthetic_markers)
        assert len(result) == 5

    def test_stationary_markers_have_small_range(self, synthetic_markers):
        from kinematic_morphospace.preprocessing.stationary import compute_marker_movement

        result = compute_marker_movement(synthetic_markers)
        stationary = result[result["marker_id"].isin([0, 1, 2])]
        assert (stationary["total_range"] < 0.01).all()

    def test_moving_markers_have_large_range(self, synthetic_markers):
        from kinematic_morphospace.preprocessing.stationary import compute_marker_movement

        result = compute_marker_movement(synthetic_markers)
        moving = result[result["marker_id"].isin([3, 4])]
        assert (moving["total_range"] > 1.0).all()


class TestDetectStationaryMarkers:
    def test_separates_stationary_from_moving(self, synthetic_markers):
        from kinematic_morphospace.preprocessing.stationary import detect_stationary_markers

        result = detect_stationary_markers(synthetic_markers)
        # Markers 0, 2 are stationary (tiny noise); 3, 4 are moving
        # Marker 1 may or may not be stationary depending on clustering
        assert bool(result[0]) is True
        assert bool(result[3]) is False
        assert bool(result[4]) is False
        # At least 2 stationary markers detected
        assert result.sum() >= 2

    def test_returns_series_indexed_by_marker_id(self, synthetic_markers):
        from kinematic_morphospace.preprocessing.stationary import detect_stationary_markers

        result = detect_stationary_markers(synthetic_markers)
        assert isinstance(result, pd.Series)
        assert set(result.index) == {0, 1, 2, 3, 4}


class TestLabelFixedObjects:
    def test_labels_by_y_position(self, synthetic_markers):
        from kinematic_morphospace.preprocessing.stationary import (
            detect_stationary_markers,
            label_fixed_objects,
        )

        is_stat = detect_stationary_markers(synthetic_markers)
        labels = label_fixed_objects(synthetic_markers, is_stat)
        # Marker 0: Y ≈ -6.7 (left_perch range -7.5 to -5.5)
        assert labels[0] == "left_perch"
        # Markers 3, 4 are moving
        assert labels[3] == "moving"
        assert labels[4] == "moving"


# ===================================================================
# trial_splitting.py tests
# ===================================================================


class TestDetectVelocityPeaks:
    def test_detects_single_flight(self):
        from kinematic_morphospace.preprocessing.trial_splitting import detect_velocity_peaks

        # Create a clear velocity peak in the middle
        n = 1000
        frames = np.arange(n)
        y = np.zeros(n, dtype=float)
        # Big Y movement in frames 400-600
        y[400:600] = np.linspace(0, 5, 200)
        df = pd.DataFrame({
            "frame": frames,
            "Y": y,
        })
        result = detect_velocity_peaks(
            df,
            min_peak_distance=50,
            min_peak_width=10,
            min_peak_height=0.001,
        )
        assert len(result) >= 1

    def test_empty_when_no_movement(self):
        from kinematic_morphospace.preprocessing.trial_splitting import detect_velocity_peaks

        df = pd.DataFrame({
            "frame": np.arange(500),
            "Y": np.zeros(500),
        })
        result = detect_velocity_peaks(df, min_peak_height=0.01)
        assert len(result) == 0


class TestSplitByTrial:
    def test_assigns_trial_numbers(self):
        from kinematic_morphospace.preprocessing.trial_splitting import split_by_trial

        df = pd.DataFrame({"frame": [0, 5, 10, 15, 20]})
        annotations = [
            {"start_frame": 3, "end_frame": 7},
            {"start_frame": 13, "end_frame": 17},
        ]
        result = split_by_trial(df, annotations)
        assert result["trial"].tolist() == [0, 1, 0, 2, 0]

    def test_no_annotations_all_zero(self):
        from kinematic_morphospace.preprocessing.trial_splitting import split_by_trial

        df = pd.DataFrame({"frame": [0, 5, 10]})
        result = split_by_trial(df, [])
        assert (result["trial"] == 0).all()


class TestAnnotationIO:
    def test_round_trip(self, tmp_path):
        from kinematic_morphospace.preprocessing.trial_splitting import (
            load_annotations,
            save_annotations,
        )

        annotations = [
            {"start_frame": 100, "end_frame": 300},
            {"start_frame": 500, "end_frame": 700},
        ]
        path = tmp_path / "trials.json"
        save_annotations(annotations, path)
        loaded = load_annotations(path)
        assert loaded == annotations


# ===================================================================
# marker_labelling.py tests
# ===================================================================


class TestComputePairwiseDistances:
    def test_basic_distances(self):
        from kinematic_morphospace.preprocessing.marker_labelling import compute_pairwise_distances

        df = pd.DataFrame({
            "frame": [0, 0, 0],
            "marker_id": [0, 1, 2],
            "X": [0.0, 1.0, 0.0],
            "Y": [0.0, 0.0, 1.0],
            "Z": [0.0, 0.0, 0.0],
        })
        result = compute_pairwise_distances(df)
        assert len(result) == 3  # C(3,2) = 3 pairs
        assert result["distance"].min() == pytest.approx(1.0)
        assert result["distance"].max() == pytest.approx(np.sqrt(2))


class TestLabelBodyMarkers:
    def test_backpack_bins_match(self, body_markers_with_distances):
        from kinematic_morphospace.preprocessing.marker_labelling import label_body_markers

        labels = label_body_markers(body_markers_with_distances)
        # Markers 10, 11, 12 should all be labelled "backpack"
        assert labels[10] == "backpack"
        assert labels[11] == "backpack"
        assert labels[12] == "backpack"

    def test_unknown_distances_unlabelled(self):
        from kinematic_morphospace.preprocessing.marker_labelling import label_body_markers

        df = pd.DataFrame({
            "frame": [0, 0],
            "marker_id": [0, 1],
            "X": [0.0, 0.5],
            "Y": [0.0, 0.0],
            "Z": [0.0, 0.0],
        })
        labels = label_body_markers(df)
        assert (labels == "unlabelled").all()


# ===================================================================
# smoothing.py tests
# ===================================================================


class TestMovingMeanSmooth:
    def test_constant_signal_unchanged(self):
        from kinematic_morphospace.preprocessing.smoothing import moving_mean_smooth

        values = np.ones(50)
        result = moving_mean_smooth(values, window=10)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_output_same_length(self):
        from kinematic_morphospace.preprocessing.smoothing import moving_mean_smooth

        values = np.random.default_rng(42).random(100)
        result = moving_mean_smooth(values, window=15)
        assert len(result) == len(values)

    def test_reduces_noise(self):
        from kinematic_morphospace.preprocessing.smoothing import moving_mean_smooth

        rng = np.random.default_rng(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = signal + rng.normal(0, 0.5, 200)
        smoothed = moving_mean_smooth(noisy, window=20)
        # Smoothed should be closer to the true signal
        assert np.std(smoothed - signal) < np.std(noisy - signal)


class TestComputeBodyStatistics:
    def test_output_columns(self, synthetic_markers):
        from kinematic_morphospace.preprocessing.smoothing import compute_body_statistics

        result = compute_body_statistics(synthetic_markers)
        for col in ["frame", "mean_X", "mean_Y", "mean_Z",
                     "smooth_X", "smooth_Y", "smooth_Z",
                     "vel_X", "vel_Y", "vel_Z", "speed"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_one_row_per_frame(self, synthetic_markers):
        from kinematic_morphospace.preprocessing.smoothing import compute_body_statistics

        result = compute_body_statistics(synthetic_markers)
        assert len(result) == synthetic_markers["frame"].nunique()


# ===================================================================
# coord_transform.py tests
# ===================================================================


class TestDetectFlightDirection:
    def test_negative_y_gives_minus_one(self, body_stats_df):
        from kinematic_morphospace.preprocessing.coord_transform import detect_flight_direction

        direction = detect_flight_direction(body_stats_df)
        assert direction == -1

    def test_positive_y_gives_plus_one(self, body_stats_df):
        from kinematic_morphospace.preprocessing.coord_transform import detect_flight_direction

        df = body_stats_df.copy()
        df["smooth_Y"] = -df["smooth_Y"]  # flip to positive Y
        direction = detect_flight_direction(df)
        assert direction == 1


class TestShiftOriginToPerch:
    def test_rightward_subtracts_right_perch(self):
        from kinematic_morphospace.preprocessing.coord_transform import RIGHT_PERCH, shift_origin_to_perch

        df = pd.DataFrame({"Y": [2.4238, 0.0, -1.0]})
        result = shift_origin_to_perch(df, direction=-1)
        np.testing.assert_allclose(result["Y"].iloc[0], 0.0, atol=1e-6)

    def test_leftward_subtracts_and_negates(self):
        from kinematic_morphospace.preprocessing.coord_transform import LEFT_PERCH, shift_origin_to_perch

        df = pd.DataFrame({"Y": [-6.658, 0.0]})
        result = shift_origin_to_perch(df, direction=1)
        np.testing.assert_allclose(result["Y"].iloc[0], 0.0, atol=1e-6)

    def test_does_not_modify_original(self):
        from kinematic_morphospace.preprocessing.coord_transform import shift_origin_to_perch

        df = pd.DataFrame({"Y": [1.0, 2.0]})
        original_y = df["Y"].copy()
        shift_origin_to_perch(df, direction=-1)
        pd.testing.assert_series_equal(df["Y"], original_y)


class TestComputeHorizontalDistance:
    def test_at_origin_is_zero(self):
        from kinematic_morphospace.preprocessing.coord_transform import compute_horizontal_distance

        df = pd.DataFrame({"smooth_X": [0.0], "smooth_Y": [0.0]})
        result = compute_horizontal_distance(df)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_pythagorean(self):
        from kinematic_morphospace.preprocessing.coord_transform import compute_horizontal_distance

        df = pd.DataFrame({"smooth_X": [3.0], "smooth_Y": [4.0]})
        result = compute_horizontal_distance(df)
        assert result.iloc[0] == pytest.approx(5.0)


# ===================================================================
# time_sync.py tests
# ===================================================================


class TestFindTakeoffFrame:
    def test_finds_first_matching_frame(self):
        from kinematic_morphospace.preprocessing.time_sync import find_takeoff_frame

        # Custom data where Y passes through the takeoff window
        df = pd.DataFrame({
            "frame": np.arange(10),
            "smooth_Y": [-9.5, -9.2, -8.9, -8.7, -8.4, -8.0, -7.5, -7.0, -6.5, -6.0],
            "speed": [0.5, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0],
        })
        result = find_takeoff_frame(df)
        assert result == 2  # first frame with Y in (-8.935, -8.5) and speed > 2

    def test_returns_none_when_no_match(self):
        from kinematic_morphospace.preprocessing.time_sync import find_takeoff_frame

        df = pd.DataFrame({
            "frame": [0, 1, 2],
            "smooth_Y": [0.0, 0.0, 0.0],
            "speed": [0.1, 0.1, 0.1],
        })
        result = find_takeoff_frame(df)
        assert result is None

    def test_finds_frame_in_range(self):
        from kinematic_morphospace.preprocessing.time_sync import find_takeoff_frame

        df = pd.DataFrame({
            "frame": [0, 1, 2, 3, 4],
            "smooth_Y": [-9.0, -8.8, -8.7, -8.3, -7.0],
            "speed": [0.5, 3.0, 4.0, 5.0, 6.0],
        })
        result = find_takeoff_frame(df)
        assert result == 1  # first frame with Y in range and speed > 2


class TestCreateTimeVariable:
    def test_time_at_frame_zero_is_zero(self):
        from kinematic_morphospace.preprocessing.time_sync import create_time_variable

        df = pd.DataFrame({"frame": [100, 200, 300]})
        result = create_time_variable(df, frame_zero=200, frame_rate=200.0)
        assert result["time"].iloc[1] == pytest.approx(0.0)

    def test_time_increments(self):
        from kinematic_morphospace.preprocessing.time_sync import create_time_variable

        df = pd.DataFrame({"frame": [0, 100, 200]})
        result = create_time_variable(df, frame_zero=0, frame_rate=100.0)
        assert result["time"].tolist() == [0.0, 1.0, 2.0]


# ===================================================================
# body_frame.py tests
# ===================================================================


class TestEstimateBodyPitch:
    def test_vertical_markers_give_zero_pitch(self):
        from kinematic_morphospace.preprocessing.body_frame import estimate_body_pitch

        # 3 markers stacked vertically → principal axis is Z → pitch ≈ 0
        df = pd.DataFrame({
            "frame": [0, 0, 0],
            "marker_id": [0, 1, 2],
            "X": [0.0, 0.0, 0.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, 0.5, 1.0],
        })
        result = estimate_body_pitch(df)
        assert len(result) == 1
        assert result["body_pitch"].iloc[0] == pytest.approx(0.0, abs=5.0)

    def test_horizontal_markers_give_ninety_pitch(self):
        from kinematic_morphospace.preprocessing.body_frame import estimate_body_pitch

        # 3 markers in X direction → principal axis is X → pitch ≈ 90°
        df = pd.DataFrame({
            "frame": [0, 0, 0],
            "marker_id": [0, 1, 2],
            "X": [0.0, 0.5, 1.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 0.0],
        })
        result = estimate_body_pitch(df)
        assert result["body_pitch"].iloc[0] == pytest.approx(90.0, abs=5.0)

    def test_too_few_markers_gives_nan(self):
        from kinematic_morphospace.preprocessing.body_frame import estimate_body_pitch

        df = pd.DataFrame({
            "frame": [0, 0],
            "marker_id": [0, 1],
            "X": [0.0, 1.0],
            "Y": [0.0, 0.0],
            "Z": [0.0, 0.0],
        })
        result = estimate_body_pitch(df, min_markers=3)
        assert pd.isna(result["body_pitch"].iloc[0])

    def test_output_columns(self):
        from kinematic_morphospace.preprocessing.body_frame import estimate_body_pitch

        df = pd.DataFrame({
            "frame": [0, 0, 0],
            "marker_id": [0, 1, 2],
            "X": [0.0, 0.5, 1.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 0.0],
        })
        result = estimate_body_pitch(df)
        for col in ["frame", "body_pitch", "normal_X", "normal_Y", "normal_Z"]:
            assert col in result.columns


# ===================================================================
# c3d_loader.py tests
# ===================================================================


class TestBuildFileList:
    def test_parses_filenames(self, tmp_path):
        from kinematic_morphospace.preprocessing.c3d_loader import build_file_list

        # Create dummy .c3d files with real filename patterns
        (tmp_path / "201130_Drogon_9m_IMUweightoff_Trial01.c3d").touch()
        (tmp_path / "201202_Ruby_9m_IMUweighton_Trial05.c3d").touch()
        (tmp_path / "201201_Charmander_9m_noIMU_Trial12.c3d").touch()
        (tmp_path / "not_a_c3d.txt").touch()

        result = build_file_list(tmp_path)
        assert len(result) == 3
        assert set(result["bird"]) == {"Drogon", "Ruby", "Charmander"}

    def test_obstacle_flag(self, tmp_path):
        from kinematic_morphospace.preprocessing.c3d_loader import build_file_list

        (tmp_path / "201203_Ruby_9m_IMUweighton_Obstacle_Trial05.c3d").touch()
        result = build_file_list(tmp_path)
        assert bool(result["obstacle"].iloc[0]) is True

    def test_nobackpack_flag(self, tmp_path):
        from kinematic_morphospace.preprocessing.c3d_loader import build_file_list

        (tmp_path / "201201_Charmander_9m_noIMU_nobackpack_Trial03.c3d").touch()
        result = build_file_list(tmp_path)
        assert bool(result["nobackpack"].iloc[0]) is True
        assert bool(result["imu"].iloc[0]) is False

    def test_bird_id_mapping(self, tmp_path):
        from kinematic_morphospace.preprocessing.c3d_loader import build_file_list

        (tmp_path / "201130_Drogon_9m_IMUweightoff_Trial01.c3d").touch()
        (tmp_path / "201202_Ruby_9m_IMUweighton_Trial02.c3d").touch()
        (tmp_path / "201204_Toothless_9m_IMUweighton_Trial03.c3d").touch()
        (tmp_path / "201201_Charmander_9m_noIMU_Trial04.c3d").touch()

        result = build_file_list(tmp_path)
        id_map = dict(zip(result["bird"], result["bird_id"]))
        assert id_map["Drogon"] == 1
        assert id_map["Ruby"] == 3
        assert id_map["Toothless"] == 4
        assert id_map["Charmander"] == 5

    def test_recursive_scan(self, tmp_path):
        from kinematic_morphospace.preprocessing.c3d_loader import build_file_list

        # Files in per-bird subdirectories
        (tmp_path / "Drogon").mkdir()
        (tmp_path / "Ruby").mkdir()
        (tmp_path / "Drogon" / "201130_Drogon_9m_IMUweightoff_Trial01.c3d").touch()
        (tmp_path / "Ruby" / "201202_Ruby_9m_IMUweighton_Trial05.c3d").touch()

        result = build_file_list(tmp_path)
        assert len(result) == 2

    def test_imu_conditions(self, tmp_path):
        from kinematic_morphospace.preprocessing.c3d_loader import build_file_list

        (tmp_path / "201130_Drogon_9m_IMUweighton_Trial01.c3d").touch()
        (tmp_path / "201130_Drogon_9m_IMUweightoff_Trial02.c3d").touch()
        (tmp_path / "201201_Drogon_9m_noIMU_Trial03.c3d").touch()

        result = build_file_list(tmp_path)
        result = result.sort_values("trial").reset_index(drop=True)
        assert bool(result["imu"].iloc[0]) is True   # IMUweighton
        assert bool(result["imu"].iloc[1]) is True   # IMUweightoff
        assert bool(result["imu"].iloc[2]) is False   # noIMU

    def test_empty_directory(self, tmp_path):
        from kinematic_morphospace.preprocessing.c3d_loader import build_file_list

        result = build_file_list(tmp_path)
        assert len(result) == 0


class TestFilterFileList:
    def test_keeps_backpack_only(self):
        from kinematic_morphospace.preprocessing.c3d_loader import filter_file_list

        df = pd.DataFrame({
            "path": ["/a.c3d", "/b.c3d", "/c.c3d"],
            "filename": ["a.c3d", "b.c3d", "c.c3d"],
            "date": ["201130", "201201", "201202"],
            "bird": ["Drogon", "Charmander", "Ruby"],
            "bird_id": [1, 5, 3],
            "distance": [9, 9, 9],
            "imu": [True, False, True],
            "obstacle": [False, False, False],
            "nobackpack": [False, True, False],
            "trial": [1, 3, 5],
        })
        result = filter_file_list(df)
        assert len(result) == 2
        assert set(result["bird"]) == {"Drogon", "Ruby"}


# ===================================================================
# c3d_pipeline.py tests
# ===================================================================


class TestC3DConfig:
    def test_defaults(self):
        from kinematic_morphospace.preprocessing.c3d_pipeline import C3DConfig

        config = C3DConfig()
        assert config.stationary_threshold == 0.001
        assert config.min_peak_distance == 250
        assert config.min_peak_width == 150
        assert config.min_peak_height == 0.01
        assert config.smooth_window == 10
        assert config.frame_rate == 200.0
        assert config.takeoff_y_range == (-8.935, -8.5)
        assert config.takeoff_min_speed == 2.0
        assert config.perch_distance == 9.0

    def test_custom_values(self):
        from kinematic_morphospace.preprocessing.c3d_pipeline import C3DConfig

        config = C3DConfig(
            stationary_threshold=0.005,
            min_peak_distance=300,
            frame_rate=100.0,
        )
        assert config.stationary_threshold == 0.005
        assert config.min_peak_distance == 300
        assert config.frame_rate == 100.0

    def test_distance_bins_defaults(self):
        from kinematic_morphospace.preprocessing.c3d_pipeline import C3DConfig
        from kinematic_morphospace.preprocessing.marker_labelling import (
            BACKPACK_BINS,
            HEADPACK_BINS,
            TAILPACK_BINS,
        )

        config = C3DConfig()
        assert config.headpack_bins == list(HEADPACK_BINS)
        assert config.backpack_bins == list(BACKPACK_BINS)
        assert config.tailpack_bins == list(TAILPACK_BINS)


# ===================================================================
# smoothing — spline tests
# ===================================================================


class TestSmoothSpline:
    def test_smooth_sine(self):
        from kinematic_morphospace.preprocessing.smoothing import smooth_spline

        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        y_smooth, velocity, acceleration = smooth_spline(x, y, rms=0.001)
        # Smoothed should be close to original for clean data
        np.testing.assert_allclose(y_smooth, y, atol=0.1)
        # Velocity should approximate cos(x)
        assert velocity is not None
        assert len(velocity) == len(x)


# ===================================================================
# Integration-style tests
# ===================================================================


class TestModuleImports:
    """Verify all new modules are importable from the package."""

    def test_import_c3d_loader(self):
        from kinematic_morphospace.preprocessing import build_file_list, filter_file_list, load_c3d

    def test_import_stationary(self):
        from kinematic_morphospace.preprocessing import (
            compute_marker_movement,
            detect_stationary_markers,
            label_fixed_objects,
        )

    def test_import_trial_splitting(self):
        from kinematic_morphospace.preprocessing import (
            detect_velocity_peaks,
            load_annotations,
            save_annotations,
            split_by_trial,
        )

    def test_import_marker_labelling(self):
        from kinematic_morphospace.preprocessing import (
            BACKPACK_BINS,
            HEADPACK_BINS,
            TAILPACK_BINS,
            compute_pairwise_distances,
            label_body_markers,
        )

    def test_import_smoothing(self):
        from kinematic_morphospace.preprocessing import (
            compute_body_statistics,
            moving_mean_smooth,
            smooth_spline,
        )

    def test_import_coord_transform(self):
        from kinematic_morphospace.preprocessing import (
            LEFT_PERCH,
            RIGHT_PERCH,
            compute_horizontal_distance,
            detect_flight_direction,
            shift_origin_all_columns,
            shift_origin_to_perch,
        )

    def test_import_time_sync(self):
        from kinematic_morphospace.preprocessing import create_time_variable, find_takeoff_frame

    def test_import_body_frame(self):
        from kinematic_morphospace.preprocessing import estimate_body_pitch

    def test_import_c3d_pipeline(self):
        from kinematic_morphospace.preprocessing import C3DConfig, run_from_c3d, run_single_c3d
