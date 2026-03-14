"""
test_data_filtering.py

Tests for the data_filtering module: filter_by() and all helper functions.
Verifies filtering by hawk name, bird ID, perch distance, obstacle, IMU,
year, turn direction, horizontal distance, and combined filters.
"""

import logging
import pytest
import numpy as np
import pandas as pd

from kinematic_morphospace.data_filtering import (
    filter_by,
    filter_by_bool,
    get_hawkID,
    filter_by_hawkname,
    filter_by_perchDist,
    filter_by_turn,
    filter_by_horzdist,
)

logger = logging.getLogger(__name__)


# -- Fixtures --

@pytest.fixture
def sample_frame_info():
    """Minimal DataFrame mimicking frame_info columns used by filter_by()."""
    return pd.DataFrame({
        'BirdID':        [1, 1, 2, 3, 4, 5, 1, 2, 3, 4],
        'PerchDistance':  [9, 9, 12, 5, 7, 9, 12, 9, 9, 5],
        'Year':          [2017, 2017, 2017, 2017, 2017, 2020, 2020, 2020, 2020, 2020],
        'Obstacle':      [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        'IMU':           [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        'Naive':         [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        'Left':          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'HorzDistance':   [-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -7.0, -5.0, -0.5, -9.0],
        'Turn':          ['Straight', 'Straight', 'Left', 'Right', 'Straight',
                          'Left', 'Right', 'Straight', 'Left', 'Right'],
    })


@pytest.fixture
def loaded_frame_info(sample_unilateraldata_path):
    """Frame info loaded from real test data via load_data + process_data."""
    from kinematic_morphospace import load_data, process_data
    data_csv = load_data(sample_unilateraldata_path)
    _, _, _, frame_info_df = process_data(data_csv)
    return frame_info_df


# -- get_hawkID tests --

class TestGetHawkID:
    def test_full_names(self):
        assert get_hawkID("Drogon") == 1
        assert get_hawkID("Rhaegal") == 2
        assert get_hawkID("Ruby") == 3
        assert get_hawkID("Toothless") == 4
        assert get_hawkID("Charmander") == 5

    def test_case_insensitive(self):
        assert get_hawkID("drogon") == 1
        assert get_hawkID("RHAEGAL") == 2

    def test_numeric_string(self):
        assert get_hawkID("1") == 1
        assert get_hawkID("5") == 5

    def test_unknown_name_returns_none(self):
        assert get_hawkID("Unknown") is None


# -- filter_by_bool tests --

class TestFilterByBool:
    def test_filters_matching_values(self):
        variable = np.array([0, 1, 0, 1, 0])
        result = filter_by_bool(variable, 1)
        np.testing.assert_array_equal(result, [False, True, False, True, False])

    def test_none_returns_all_true(self):
        variable = np.array([0, 1, 0])
        result = filter_by_bool(variable, None)
        np.testing.assert_array_equal(result, [True, True, True])


# -- filter_by_hawkname tests --

class TestFilterByHawkname:
    def test_filters_correct_bird(self, sample_frame_info):
        birdID = sample_frame_info['BirdID']
        mask = filter_by_hawkname(birdID, "Drogon")
        # BirdID 1 (Drogon) at indices 0, 1, 6
        assert mask.sum() == 3
        assert list(mask[mask].index) == [0, 1, 6]

    def test_none_returns_all(self, sample_frame_info):
        birdID = sample_frame_info['BirdID']
        mask = filter_by_hawkname(birdID, None)
        assert mask.all()

    def test_unknown_name_returns_all(self, sample_frame_info):
        birdID = sample_frame_info['BirdID']
        mask = filter_by_hawkname(birdID, "Unknown")
        assert mask.all()


# -- filter_by_perchDist tests --

class TestFilterByPerchDist:
    def test_integer_value(self, sample_frame_info):
        perchDist = sample_frame_info['PerchDistance']
        mask = filter_by_perchDist(perchDist, 9)
        expected_indices = [0, 1, 5, 7, 8]
        assert mask.sum() == 5
        assert list(mask[mask].index) == expected_indices

    def test_string_value(self, sample_frame_info):
        perchDist = sample_frame_info['PerchDistance']
        mask = filter_by_perchDist(perchDist, '12m')
        assert mask.sum() == 2
        assert list(mask[mask].index) == [2, 6]

    def test_list_of_distances(self, sample_frame_info):
        perchDist = sample_frame_info['PerchDistance']
        mask = filter_by_perchDist(perchDist, [9, 12])
        assert mask.sum() == 7

    def test_none_returns_all(self, sample_frame_info):
        perchDist = sample_frame_info['PerchDistance']
        mask = filter_by_perchDist(perchDist, None)
        assert mask.all()


# -- filter_by_turn tests --

class TestFilterByTurn:
    def test_left_turn(self, sample_frame_info):
        turn = sample_frame_info['Turn']
        mask = filter_by_turn(turn, 'left')
        assert mask.sum() == 3

    def test_right_turn(self, sample_frame_info):
        turn = sample_frame_info['Turn']
        mask = filter_by_turn(turn, 'right')
        assert mask.sum() == 3

    def test_straight(self, sample_frame_info):
        turn = sample_frame_info['Turn']
        mask = filter_by_turn(turn, 'straight')
        assert mask.sum() == 4

    def test_none_returns_all(self, sample_frame_info):
        turn = sample_frame_info['Turn']
        mask = filter_by_turn(turn, None)
        assert mask.all()


# -- filter_by_horzdist tests --

class TestFilterByHorzdist:
    def test_single_value(self, sample_frame_info):
        horzdist = sample_frame_info['HorzDistance']
        # Should return x < -4.5
        mask = filter_by_horzdist(horzdist, 4.5)
        expected = horzdist < -4.5
        np.testing.assert_array_equal(mask, expected)

    def test_positive_value_converted_to_negative(self, sample_frame_info):
        horzdist = sample_frame_info['HorzDistance']
        mask_pos = filter_by_horzdist(horzdist, 4.5)
        mask_neg = filter_by_horzdist(horzdist, -4.5)
        np.testing.assert_array_equal(mask_pos, mask_neg)

    def test_tuple_range(self, sample_frame_info):
        horzdist = sample_frame_info['HorzDistance']
        # Range (6, 2) means -6 <= x < -2
        mask = filter_by_horzdist(horzdist, (6, 2))
        expected = (horzdist >= -6) & (horzdist < -2)
        np.testing.assert_array_equal(mask, expected)

    def test_keyword_first_half(self, sample_frame_info):
        horzdist = sample_frame_info['HorzDistance']
        mask = filter_by_horzdist(horzdist, 'first_half')
        expected = horzdist < -4.5
        np.testing.assert_array_equal(mask, expected)

    def test_keyword_landing(self, sample_frame_info):
        horzdist = sample_frame_info['HorzDistance']
        mask = filter_by_horzdist(horzdist, 'landing')
        expected = (horzdist >= -2) & (horzdist < 0)
        np.testing.assert_array_equal(mask, expected)

    def test_unknown_keyword_raises(self, sample_frame_info):
        horzdist = sample_frame_info['HorzDistance']
        with pytest.raises(ValueError, match="Unknown keyword"):
            filter_by_horzdist(horzdist, 'invalid')

    def test_none_returns_all(self, sample_frame_info):
        horzdist = sample_frame_info['HorzDistance']
        mask = filter_by_horzdist(horzdist, None)
        assert mask.all()


# -- filter_by (main function) tests --

class TestFilterBy:
    def test_single_hawkname_filter(self, sample_frame_info):
        mask = filter_by(sample_frame_info, hawkname="Ruby")
        assert mask.sum() == 2  # BirdID 3 at indices 3, 8

    def test_single_birdID_filter(self, sample_frame_info):
        mask = filter_by(sample_frame_info, birdID=2)
        assert mask.sum() == 2  # indices 2, 7

    def test_year_filter(self, sample_frame_info):
        mask = filter_by(sample_frame_info, year=2017)
        assert mask.sum() == 5

    def test_obstacle_filter(self, sample_frame_info):
        mask = filter_by(sample_frame_info, obstacle=1)
        assert mask.sum() == 3

    def test_imu_filter(self, sample_frame_info):
        mask = filter_by(sample_frame_info, IMU=1)
        assert mask.sum() == 3

    def test_combined_filters(self, sample_frame_info):
        mask = filter_by(sample_frame_info, year=2020, obstacle=0)
        # Year 2020: indices 5,6,7,8,9; obstacle=0: indices 7,9
        assert mask.sum() == 2

    def test_no_filters_returns_all_true(self, sample_frame_info):
        mask = filter_by(sample_frame_info)
        assert mask.all()
        assert len(mask) == len(sample_frame_info)

    def test_unrecognised_filter_warns(self, sample_frame_info, capsys):
        mask = filter_by(sample_frame_info, invalid_filter=True)
        captured = capsys.readouterr()
        assert "unrecognised filters" in captured.out.lower()
        # Should still return a valid mask (all True since no valid filter applied)
        assert mask.all()

    def test_on_real_data(self, loaded_frame_info):
        """Verify filter_by works on real test data loaded from CSV."""
        mask = filter_by(loaded_frame_info, year=2020)
        assert hasattr(mask, 'dtype')
        assert mask.dtype == bool
        assert mask.sum() > 0
        assert mask.sum() < len(loaded_frame_info)

    def test_hawkname_and_year_on_real_data(self, loaded_frame_info):
        """Combined filters on real data should produce a strict subset."""
        all_drogon = filter_by(loaded_frame_info, hawkname="Drogon")
        drogon_2017 = filter_by(loaded_frame_info, hawkname="Drogon", year=2017)
        assert drogon_2017.sum() <= all_drogon.sum()
