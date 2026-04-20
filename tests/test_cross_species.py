"""Tests for kinematic_morphospace.cross_species — Harvey et al. cross-species data pipeline."""

import numpy as np
import pandas as pd
import pytest

from kinematic_morphospace.cross_species import (
    clean_body_data,
    check_and_fix_shoulder_distance,
    compute_derived_markers,
    fill_missing_body_measurements,
    filter_marker_columns,
    fix_leftright_sign,
    integrate_dataframe_to_bird3D,
    load_harvey_data,
    merge_bird_data,
    mirror_marker,
    process_body_bird_id,
    select_best_specimen_per_species,
    select_max_wingspan_row,
    set_new_origin_and_axes,
    split_bird_id,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic Harvey-like DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_wing_csv(tmp_path):
    """Create a temporary CSV file mimicking Harvey wing data."""
    df = pd.DataFrame({
        'BirdID': ['bird_01', 'bird_01', 'bird_02', 'bird_02'],
        'pt8_X': [0.1, 0.15, 0.2, 0.25],
        'pt8_Y': [0.0, 0.0, 0.0, 0.0],
        'pt8_Z': [0.0, 0.0, 0.0, 0.0],
        'pt12_X': [-0.1, -0.15, -0.2, -0.25],
        'pt12_Y': [0.0, 0.0, 0.0, 0.0],
        'pt12_Z': [0.0, 0.0, 0.0, 0.0],
    })
    path = tmp_path / "wing.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_body_csv(tmp_path):
    """Create a temporary CSV file mimicking Harvey body data."""
    df = pd.DataFrame({
        'bird_id': ['barn_owl_bird_01', 'COLLI_bird_02', None],
        'species_common': ['Barn owl', 'Pigeon', 'Unknown'],
        'x_loc_of_body_max_cm': [5.0, 4.0, 3.0],
        'x_loc_of_humeral_insert_cm': [2.0, 1.5, 1.0],
        'y_loc_of_humeral_insert_cm': [1.0, 0.8, 0.6],
        'z_loc_of_humeral_insert_cm': [0.5, 0.4, 0.3],
        'body_width_max_cm': [10.0, 8.0, 6.0],
        'width_at_leg_insert_cm': [6.0, 5.0, 4.0],
        'head_length_cm': [5.0, 4.0, 3.0],
        'neck_length_cm': [6.0, 5.0, 4.0],
        'body_length_cm': [20.0, 15.0, 12.0],
        'wing_span_cm': [106.0, 63.4, 50.0],
        'tail_width_cm': [8.0, 6.0, 5.0],
        'tail_length_cm': [12.0, 10.0, 8.0],
        'torsotail_length_cm': [25.0, 20.0, 15.0],
    })
    path = tmp_path / "body.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_wing_df():
    """Synthetic wing DataFrame with pt marker columns."""
    return pd.DataFrame({
        'BirdID': ['bird_01', 'bird_01', 'bird_02', 'bird_02'],
        'pt1_X': [0.01, 0.02, 0.03, 0.04],
        'pt1_Y': [0.0, 0.0, 0.0, 0.0],
        'pt1_Z': [0.0, 0.0, 0.0, 0.0],
        'pt2_X': [0.05, 0.06, 0.07, 0.08],
        'pt2_Y': [0.0, 0.0, 0.0, 0.0],
        'pt2_Z': [0.01, 0.01, 0.01, 0.01],
        'pt4_X': [0.1, 0.12, 0.14, 0.16],
        'pt4_Y': [0.05, 0.06, 0.07, 0.08],
        'pt4_Z': [0.0, 0.0, 0.0, 0.0],
        'pt8_X': [0.2, 0.25, 0.3, 0.35],
        'pt8_Y': [0.1, 0.12, 0.14, 0.16],
        'pt8_Z': [0.0, 0.0, 0.0, 0.0],
        'pt9_X': [0.3, 0.35, 0.4, 0.45],
        'pt9_Y': [0.15, 0.18, 0.2, 0.22],
        'pt9_Z': [0.0, 0.0, 0.0, 0.0],
        'pt10_X': [0.15, 0.18, 0.2, 0.22],
        'pt10_Y': [0.08, 0.09, 0.1, 0.11],
        'pt10_Z': [0.0, 0.0, 0.0, 0.0],
        'pt11_X': [0.0, 0.0, 0.0, 0.0],
        'pt11_Y': [-0.1, -0.1, -0.1, -0.1],
        'pt11_Z': [0.0, 0.0, 0.0, 0.0],
        'pt12_X': [-0.2, -0.25, -0.3, -0.35],
        'pt12_Y': [0.1, 0.12, 0.14, 0.16],
        'pt12_Z': [0.0, 0.0, 0.0, 0.0],
    })


@pytest.fixture
def sample_body_df():
    """Synthetic body DataFrame with bird_id and measurements."""
    return pd.DataFrame({
        'bird_id': ['barn_owl_bird_01', 'COLLI_bird_02'],
        'species_common': ['Barn owl', 'Pigeon'],
        'x_loc_of_body_max_cm': [5.0, 4.0],
        'x_loc_of_humeral_insert_cm': [2.0, 1.5],
        'y_loc_of_humeral_insert_cm': [1.0, 0.8],
        'z_loc_of_humeral_insert_cm': [0.5, 0.4],
        'body_width_max_cm': [10.0, 8.0],
        'width_at_leg_insert_cm': [6.0, 5.0],
        'head_length_cm': [5.0, 4.0],
        'neck_length_cm': [6.0, 5.0],
        'body_length_cm': [20.0, 15.0],
        'wing_span_cm': [106.0, 63.4],
        'tail_width_cm': [8.0, 6.0],
        'tail_length_cm': [12.0, 10.0],
        'torsotail_length_cm': [25.0, 20.0],
    })


@pytest.fixture
def merged_df_with_markers(sample_wing_df):
    """DataFrame with pt markers plus body measurements — ready for derive/fix steps."""
    df = sample_wing_df.copy()
    df['body_width_max_cm'] = [10.0, 10.0, 12.0, 12.0]
    df['width_at_leg_insert_cm'] = [6.0, 6.0, 7.0, 7.0]
    df['head_length_cm'] = [5.0, 5.0, 6.0, 6.0]
    df['neck_length_cm'] = [6.0, 6.0, 7.0, 7.0]
    df['tail_length_cm'] = [12.0, 12.0, 14.0, 14.0]
    df['tail_width_cm'] = [8.0, 8.0, 9.0, 9.0]
    df['wing_span_cm'] = [60.0, 60.0, 70.0, 70.0]
    df['species_common'] = ['Barn owl', 'Barn owl', 'Pigeon', 'Pigeon']
    return df


# ---------------------------------------------------------------------------
# Tests: load_harvey_data
# ---------------------------------------------------------------------------

class TestLoadHarveyData:
    def test_returns_two_dataframes(self, sample_wing_csv, sample_body_csv):
        wing_df, body_df = load_harvey_data(sample_wing_csv, sample_body_csv)
        assert isinstance(wing_df, pd.DataFrame)
        assert isinstance(body_df, pd.DataFrame)

    def test_loads_correct_shape(self, sample_wing_csv, sample_body_csv):
        wing_df, body_df = load_harvey_data(sample_wing_csv, sample_body_csv)
        assert len(wing_df) == 4
        assert len(body_df) == 3


# ---------------------------------------------------------------------------
# Tests: select_max_wingspan_row
# ---------------------------------------------------------------------------

class TestSelectMaxWingspanRow:
    def test_selects_max_wingspan_per_bird(self):
        df = pd.DataFrame({
            'BirdID': ['A', 'A', 'B', 'B'],
            'pt8_X': [0.1, 0.2, 0.3, 0.15],
            'pt8_Y': [0.0, 0.0, 0.0, 0.0],
            'pt8_Z': [0.0, 0.0, 0.0, 0.0],
            'pt12_X': [-0.1, -0.2, -0.3, -0.15],
            'pt12_Y': [0.0, 0.0, 0.0, 0.0],
            'pt12_Z': [0.0, 0.0, 0.0, 0.0],
        })
        result = select_max_wingspan_row(df)
        assert len(result) == 2
        # Bird A: max wingspan at row 1 (0.4 span), Bird B: at row 2 (0.6 span)
        assert result.iloc[0]['pt8_X'] == pytest.approx(0.2)
        assert result.iloc[1]['pt8_X'] == pytest.approx(0.3)

    def test_single_bird(self):
        df = pd.DataFrame({
            'BirdID': ['A', 'A'],
            'pt8_X': [0.1, 0.2],
            'pt8_Y': [0.0, 0.0],
            'pt8_Z': [0.0, 0.0],
            'pt12_X': [-0.1, -0.2],
            'pt12_Y': [0.0, 0.0],
            'pt12_Z': [0.0, 0.0],
        })
        result = select_max_wingspan_row(df)
        assert len(result) == 1

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            'BirdID': ['A', 'A'],
            'pt8_X': [0.1, 0.2],
            'pt8_Y': [0.0, 0.0],
            'pt8_Z': [0.0, 0.0],
            'pt12_X': [-0.1, -0.2],
            'pt12_Y': [0.0, 0.0],
            'pt12_Z': [0.0, 0.0],
        })
        original_cols = set(df.columns)
        _ = select_max_wingspan_row(df)
        # The original df should not have a 'wingspan' column added
        assert set(df.columns) == original_cols, \
            "select_max_wingspan_row should not mutate the input DataFrame"

    def test_custom_markers(self):
        df = pd.DataFrame({
            'BirdID': ['A'],
            'left_X': [0.5], 'left_Y': [0.0], 'left_Z': [0.0],
            'right_X': [-0.5], 'right_Y': [0.0], 'right_Z': [0.0],
        })
        result = select_max_wingspan_row(df, left_marker='left', right_marker='right')
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: select_best_specimen_per_species
# ---------------------------------------------------------------------------

class TestSelectBestSpecimenPerSpecies:
    """Tests for select_best_specimen_per_species on a merged DataFrame."""

    def _make_df(self, records):
        """Build a minimal merged DataFrame from a list of dicts."""
        defaults = {
            'body_width_max_cm': 10.0,
            'width_at_leg_insert_cm': 5.0,
            'tail_width_cm': 4.0,
            'tail_length_cm': 8.0,
            'wing_span_cm': 60.0,
        }
        rows = []
        for r in records:
            row = dict(defaults)
            row.update(r)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_one_row_per_species_returned(self):
        df = self._make_df([
            {'species_common': 'hawk', 'wing_span_cm': 60.0},
            {'species_common': 'hawk', 'wing_span_cm': 70.0},
            {'species_common': 'pigeon', 'wing_span_cm': 55.0},
        ])
        result = select_best_specimen_per_species(df)
        assert len(result) == 2
        species_set = set(result['species_common'])
        assert species_set == {'hawk', 'pigeon'}

    def test_prefers_complete_over_large_wingspan(self):
        """A specimen with all body measurements should be preferred even if
        another specimen in the same species has a larger wingspan but NaN values."""
        df = self._make_df([
            {
                'species_common': 'stellers_jay',
                'wing_span_cm': 90.0,
                'body_width_max_cm': float('nan'),   # incomplete
                'tail_length_cm': float('nan'),
            },
            {
                'species_common': 'stellers_jay',
                'wing_span_cm': 70.0,                # smaller wingspan but complete
            },
        ])
        result = select_best_specimen_per_species(df)
        assert len(result) == 1
        assert result.iloc[0]['wing_span_cm'] == pytest.approx(70.0)

    def test_picks_largest_wingspan_among_complete_specimens(self):
        """When multiple specimens are all complete, select the one with the largest wingspan."""
        df = self._make_df([
            {'species_common': 'hawk', 'wing_span_cm': 60.0},
            {'species_common': 'hawk', 'wing_span_cm': 80.0},
            {'species_common': 'hawk', 'wing_span_cm': 70.0},
        ])
        result = select_best_specimen_per_species(df)
        assert len(result) == 1
        assert result.iloc[0]['wing_span_cm'] == pytest.approx(80.0)

    def test_single_specimen_per_species_unchanged(self):
        """When there is already one specimen per species, the result is unchanged."""
        df = self._make_df([
            {'species_common': 'hawk', 'wing_span_cm': 60.0},
            {'species_common': 'pigeon', 'wing_span_cm': 55.0},
        ])
        result = select_best_specimen_per_species(df)
        assert len(result) == 2

    def test_no_internal_nan_count_column_in_output(self):
        """The temporary _nan_count helper column must not appear in the result."""
        df = self._make_df([
            {'species_common': 'hawk', 'wing_span_cm': 60.0},
        ])
        result = select_best_specimen_per_species(df)
        assert '_nan_count' not in result.columns

    def test_does_not_modify_original(self):
        df = self._make_df([
            {'species_common': 'hawk', 'wing_span_cm': 60.0},
            {'species_common': 'hawk', 'wing_span_cm': 70.0},
        ])
        original_cols = set(df.columns)
        original_len = len(df)
        _ = select_best_specimen_per_species(df)
        assert set(df.columns) == original_cols, \
            "select_best_specimen_per_species should not mutate the input DataFrame"
        assert len(df) == original_len

    def test_custom_completeness_columns(self):
        """A custom completeness_cols list is respected."""
        df = self._make_df([
            {
                'species_common': 'hawk',
                'wing_span_cm': 90.0,
                'body_width_max_cm': float('nan'),   # incomplete by default cols
            },
            {
                'species_common': 'hawk',
                'wing_span_cm': 60.0,
            },
        ])
        # Using only tail_length_cm for completeness — both rows are complete on it
        result = select_best_specimen_per_species(
            df, completeness_cols=['tail_length_cm']
        )
        # Both rows complete on the custom column → max wingspan wins
        assert result.iloc[0]['wing_span_cm'] == pytest.approx(90.0)

    def test_partial_completeness_cols_missing_from_df(self):
        """If a completeness column is not present in the DataFrame, it is silently ignored."""
        df = self._make_df([
            {'species_common': 'hawk', 'wing_span_cm': 75.0},
        ])
        # 'nonexistent_col' does not exist — should not raise
        result = select_best_specimen_per_species(
            df, completeness_cols=['nonexistent_col', 'tail_length_cm']
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: clean_body_data
# ---------------------------------------------------------------------------

class TestCleanBodyData:
    def test_returns_only_expected_columns(self, sample_body_df):
        result = clean_body_data(sample_body_df)
        expected_cols = {
            'bird_id', 'species_common', 'x_loc_of_body_max_cm',
            'x_loc_of_humeral_insert_cm', 'y_loc_of_humeral_insert_cm',
            'z_loc_of_humeral_insert_cm', 'body_width_max_cm',
            'width_at_leg_insert_cm', 'head_length_cm', 'neck_length_cm',
            'body_length_cm', 'wing_span_cm', 'tail_width_cm', 'tail_length_cm',
            'torsotail_length_cm',
        }
        assert set(result.columns) == expected_cols

    def test_preserves_row_count(self, sample_body_df):
        result = clean_body_data(sample_body_df)
        assert len(result) == len(sample_body_df)


# ---------------------------------------------------------------------------
# Tests: split_bird_id
# ---------------------------------------------------------------------------

class TestSplitBirdId:
    def test_standard_split(self):
        result = split_bird_id('barn_owl_bird_01')
        assert result['Species'] == 'barn_owl'
        assert result['BirdID'] == 'bird_01'

    def test_single_part_species(self):
        result = split_bird_id('COLLI_bird_02')
        assert result['Species'] == 'COLLI'
        assert result['BirdID'] == 'bird_02'

    def test_multi_part_species(self):
        result = split_bird_id('great_blue_heron_bird_03')
        assert result['Species'] == 'great_blue_heron'
        assert result['BirdID'] == 'bird_03'


# ---------------------------------------------------------------------------
# Tests: process_body_bird_id
# ---------------------------------------------------------------------------

class TestProcessBodyBirdId:
    def test_filters_nan_rows(self):
        df = pd.DataFrame({
            'bird_id': ['barn_owl_bird_01', None, 'COLLI_bird_02'],
            'species_common': ['Barn owl', 'Unknown', 'Pigeon'],
        })
        result = process_body_bird_id(df)
        assert len(result) == 2

    def test_adds_species_and_birdid_columns(self, sample_body_df):
        result = process_body_bird_id(sample_body_df)
        assert 'Species' in result.columns
        assert 'BirdID' in result.columns

    def test_replaces_COLLI_with_col_liv(self, sample_body_df):
        result = process_body_bird_id(sample_body_df)
        species_vals = result['Species'].tolist()
        assert 'col_liv' in species_vals
        # Also check it's lowercase
        for s in species_vals:
            assert s == s.lower()


# ---------------------------------------------------------------------------
# Tests: merge_bird_data
# ---------------------------------------------------------------------------

class TestMergeBirdData:
    def test_left_join_preserves_wing_rows(self):
        wing_df = pd.DataFrame({'BirdID': ['A', 'B', 'C'], 'val': [1, 2, 3]})
        body_df = pd.DataFrame({'BirdID': ['A', 'B'], 'mass': [10, 20]})
        result = merge_bird_data(wing_df, body_df)
        assert len(result) == 3
        assert pd.isna(result.loc[result['BirdID'] == 'C', 'mass'].iloc[0])

    def test_merges_on_custom_column(self):
        wing_df = pd.DataFrame({'id': ['A', 'B'], 'val': [1, 2]})
        body_df = pd.DataFrame({'id': ['A', 'B'], 'mass': [10, 20]})
        result = merge_bird_data(wing_df, body_df, on_col='id')
        assert 'mass' in result.columns


# ---------------------------------------------------------------------------
# Tests: filter_marker_columns
# ---------------------------------------------------------------------------

class TestFilterMarkerColumns:
    def test_filters_correct_columns(self):
        df = pd.DataFrame({
            'BirdID': ['A'],
            'pt8_X': [1.0], 'pt8_Y': [2.0], 'pt8_Z': [3.0],
            'pt12_X': [4.0], 'pt12_Y': [5.0], 'pt12_Z': [6.0],
            'other_col': [99],
        })
        result = filter_marker_columns(df, ['pt8', 'pt12'], ['BirdID'])
        assert 'BirdID' in result.columns
        assert 'pt8_X' in result.columns
        assert 'other_col' not in result.columns

    def test_returns_base_columns_first(self):
        df = pd.DataFrame({
            'BirdID': ['A'],
            'pt1_X': [1.0],
            'species': ['hawk'],
        })
        result = filter_marker_columns(df, ['pt1'], ['BirdID', 'species'])
        assert list(result.columns[:2]) == ['BirdID', 'species']


# ---------------------------------------------------------------------------
# Tests: set_new_origin_and_axes
# ---------------------------------------------------------------------------

class TestSetNewOriginAndAxes:
    def test_single_marker_origin(self):
        """Origin from a single marker shifts all pt coordinates."""
        df = pd.DataFrame({
            'pt1_X': [1.0], 'pt1_Y': [2.0], 'pt1_Z': [3.0],
            'pt2_X': [4.0], 'pt2_Y': [5.0], 'pt2_Z': [6.0],
        })
        result = set_new_origin_and_axes(
            df, origin_marker='pt1',
            origin_axes=('x', 'y', 'z'),
            new_axes=('x', 'y', 'z'),
        )
        # pt1 should be at origin
        assert result['pt1_X'].iloc[0] == pytest.approx(0.0)
        assert result['pt1_Y'].iloc[0] == pytest.approx(0.0)
        assert result['pt1_Z'].iloc[0] == pytest.approx(0.0)
        # pt2 should be shifted by pt1's position
        assert result['pt2_X'].iloc[0] == pytest.approx(3.0)
        assert result['pt2_Y'].iloc[0] == pytest.approx(3.0)
        assert result['pt2_Z'].iloc[0] == pytest.approx(3.0)

    def test_dual_marker_origin_averages(self):
        """When origin_marker is a list, origin is the average of two markers."""
        df = pd.DataFrame({
            'pt1_X': [2.0], 'pt1_Y': [0.0], 'pt1_Z': [0.0],
            'pt2_X': [4.0], 'pt2_Y': [0.0], 'pt2_Z': [0.0],
        })
        result = set_new_origin_and_axes(
            df, origin_marker=['pt1', 'pt2'],
            origin_axes=('x', 'y', 'z'),
            new_axes=('x', 'y', 'z'),
        )
        # Average origin x = (2+4)/2 = 3.0
        # pt1 should be at -1.0, pt2 at +1.0
        assert result['pt1_X'].iloc[0] == pytest.approx(-1.0)
        assert result['pt2_X'].iloc[0] == pytest.approx(1.0)

    def test_axis_remapping(self):
        """Test axis remapping with new_axes=('y', '-x', 'z')."""
        df = pd.DataFrame({
            'pt1_X': [1.0], 'pt1_Y': [2.0], 'pt1_Z': [3.0],
        })
        result = set_new_origin_and_axes(
            df, origin_marker='pt1',
            origin_axes=('x', 'y', 'z'),
            new_axes=('y', '-x', 'z'),
        )
        # After origin subtraction: (0, 0, 0) -> remapping doesn't matter for origin point
        assert result['pt1_X'].iloc[0] == pytest.approx(0.0)
        assert result['pt1_Y'].iloc[0] == pytest.approx(0.0)
        assert result['pt1_Z'].iloc[0] == pytest.approx(0.0)

    def test_axis_remapping_nonzero(self):
        """Test axis remapping on a non-origin point."""
        df = pd.DataFrame({
            'pt1_X': [0.0], 'pt1_Y': [0.0], 'pt1_Z': [0.0],
            'pt2_X': [1.0], 'pt2_Y': [2.0], 'pt2_Z': [3.0],
        })
        result = set_new_origin_and_axes(
            df, origin_marker='pt1',
            origin_axes=('x', 'y', 'z'),
            new_axes=('y', '-x', 'z'),
        )
        # pt2 original relative coords: (1, 2, 3)
        # new X = orig_y = 2, new Y = -orig_x = -1, new Z = orig_z = 3
        assert result['pt2_X'].iloc[0] == pytest.approx(2.0)
        assert result['pt2_Y'].iloc[0] == pytest.approx(-1.0)
        assert result['pt2_Z'].iloc[0] == pytest.approx(3.0)

    def test_does_not_modify_original(self, sample_wing_df):
        original = sample_wing_df.copy()
        _ = set_new_origin_and_axes(sample_wing_df, origin_marker='pt2')
        pd.testing.assert_frame_equal(sample_wing_df, original)


# ---------------------------------------------------------------------------
# Tests: mirror_marker
# ---------------------------------------------------------------------------

class TestMirrorMarker:
    def test_mirrors_x_negated_for_left(self):
        df = pd.DataFrame({'src_x': [1.0], 'src_y': [2.0], 'src_z': [3.0]})
        mirror_marker(df, 'right_wing', 'left_wing', 'src_x', 'src_y', 'src_z')
        assert df['right_wing_x'].iloc[0] == pytest.approx(1.0)
        assert df['left_wing_x'].iloc[0] == pytest.approx(-1.0)

    def test_preserves_y_and_z(self):
        df = pd.DataFrame({'src_x': [1.0], 'src_y': [2.0], 'src_z': [3.0]})
        mirror_marker(df, 'right_wing', 'left_wing', 'src_x', 'src_y', 'src_z')
        assert df['right_wing_y'].iloc[0] == pytest.approx(2.0)
        assert df['left_wing_y'].iloc[0] == pytest.approx(2.0)
        assert df['right_wing_z'].iloc[0] == pytest.approx(3.0)
        assert df['left_wing_z'].iloc[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Tests: compute_derived_markers
# ---------------------------------------------------------------------------

class TestComputeDerivedMarkers:
    def test_creates_wingtip_markers(self, merged_df_with_markers):
        result = compute_derived_markers(merged_df_with_markers)
        assert 'right_wingtip_x' in result.columns
        assert 'left_wingtip_x' in result.columns
        # Right wingtip comes from pt9_X, left should be negated
        assert result['right_wingtip_x'].iloc[0] == pytest.approx(
            merged_df_with_markers['pt9_X'].iloc[0])
        assert result['left_wingtip_x'].iloc[0] == pytest.approx(
            -merged_df_with_markers['pt9_X'].iloc[0])

    def test_creates_primary_markers_from_average(self, merged_df_with_markers):
        result = compute_derived_markers(merged_df_with_markers)
        expected_x = (merged_df_with_markers['pt8_X'].iloc[0] +
                      merged_df_with_markers['pt4_X'].iloc[0]) / 2
        assert result['right_primary_x'].iloc[0] == pytest.approx(expected_x)

    def test_creates_hood_marker(self, merged_df_with_markers):
        result = compute_derived_markers(merged_df_with_markers)
        assert 'hood_x' in result.columns
        assert result['hood_x'].iloc[0] == pytest.approx(0.0)
        # hood_y = (neck_length + head_length) / 100
        expected_hood_y = (merged_df_with_markers['neck_length_cm'].iloc[0] +
                          merged_df_with_markers['head_length_cm'].iloc[0]) / 100
        assert result['hood_y'].iloc[0] == pytest.approx(expected_hood_y)

    def test_drops_temporary_columns(self, merged_df_with_markers):
        result = compute_derived_markers(merged_df_with_markers)
        for col in ['primary_avg_x', 'primary_avg_y', 'primary_avg_z',
                     'tailtip_x', 'tailtip_y', 'tailtip_z']:
            assert col not in result.columns

    def test_does_not_modify_original(self, merged_df_with_markers):
        original = merged_df_with_markers.copy()
        _ = compute_derived_markers(merged_df_with_markers)
        pd.testing.assert_frame_equal(merged_df_with_markers, original)


# ---------------------------------------------------------------------------
# Tests: fix_leftright_sign
# ---------------------------------------------------------------------------

class TestFixLeftrightSign:
    def test_corrects_swapped_signs(self):
        """When left is positive and right is negative, they should be corrected."""
        df = pd.DataFrame({
            'left_shoulder_x': [0.1],   # wrong: should be negative
            'right_shoulder_x': [-0.1],  # wrong: should be positive
        })
        result = fix_leftright_sign(df)
        assert result['left_shoulder_x'].iloc[0] < 0
        assert result['right_shoulder_x'].iloc[0] > 0

    def test_no_change_when_correct(self):
        """When signs are already correct, no change."""
        df = pd.DataFrame({
            'left_shoulder_x': [-0.1],
            'right_shoulder_x': [0.1],
        })
        result = fix_leftright_sign(df)
        assert result['left_shoulder_x'].iloc[0] == pytest.approx(-0.1)
        assert result['right_shoulder_x'].iloc[0] == pytest.approx(0.1)

    def test_skips_missing_marker_columns(self):
        """Should not raise when some marker columns are missing."""
        df = pd.DataFrame({
            'left_shoulder_x': [-0.1],
            'right_shoulder_x': [0.1],
            # No wingtip, primary, etc.
        })
        result = fix_leftright_sign(df)
        assert len(result) == 1

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            'left_shoulder_x': [0.1],
            'right_shoulder_x': [-0.1],
        })
        original = df.copy()
        _ = fix_leftright_sign(df)
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# Tests: check_and_fix_shoulder_distance
# ---------------------------------------------------------------------------

class TestCheckAndFixShoulderDistance:
    @pytest.fixture
    def df_with_bilateral_markers(self):
        """DataFrame with bilateral markers and body measurements for shoulder adjustment."""
        return pd.DataFrame({
            # Shoulder markers — distance = 0.2 (±0.1)
            'left_shoulder_x': [-0.1], 'left_shoulder_y': [0.0], 'left_shoulder_z': [0.01],
            'right_shoulder_x': [0.1], 'right_shoulder_y': [0.0], 'right_shoulder_z': [0.01],
            # Wingtip markers
            'left_wingtip_x': [-0.3], 'left_wingtip_y': [0.1], 'left_wingtip_z': [0.0],
            'right_wingtip_x': [0.3], 'right_wingtip_y': [0.1], 'right_wingtip_z': [0.0],
            # Primary markers
            'left_primary_x': [-0.2], 'left_primary_y': [0.08], 'left_primary_z': [0.0],
            'right_primary_x': [0.2], 'right_primary_y': [0.08], 'right_primary_z': [0.0],
            # Secondary markers
            'left_secondary_x': [-0.15], 'left_secondary_y': [0.05], 'left_secondary_z': [0.01],
            'right_secondary_x': [0.15], 'right_secondary_y': [0.05], 'right_secondary_z': [0.01],
            # Tailtip markers
            'left_tailtip_x': [-0.05], 'left_tailtip_y': [-0.1], 'left_tailtip_z': [0.0],
            'right_tailtip_x': [0.05], 'right_tailtip_y': [-0.1], 'right_tailtip_z': [0.0],
            # Tailbase markers
            'left_tailbase_x': [-0.03], 'left_tailbase_y': [-0.05], 'left_tailbase_z': [0.01],
            'right_tailbase_x': [0.03], 'right_tailbase_y': [-0.05], 'right_tailbase_z': [0.01],
            # Body measurements
            'body_width_max_cm': [20.0],
            'y_loc_of_humeral_insert_cm': [10.0],  # expected shoulder dist = 0.2m = matches
            'width_at_leg_insert_cm': [3.0],
            'tail_width_cm': [5.0],
            'wing_span_cm': [60.0],
            # pt markers needed for shoulder width adjustment
            'pt1_X': [0.01], 'pt2_X': [0.05],
        })

    def test_no_adjustment_within_tolerance(self, df_with_bilateral_markers):
        """When distances match expectations, no adjustment needed."""
        result = check_and_fix_shoulder_distance(df_with_bilateral_markers, tolerance=0.5)
        # With high tolerance, should see minimal changes
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_returns_dataframe(self, df_with_bilateral_markers):
        result = check_and_fix_shoulder_distance(df_with_bilateral_markers)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_original(self, df_with_bilateral_markers):
        original = df_with_bilateral_markers.copy()
        _ = check_and_fix_shoulder_distance(df_with_bilateral_markers)
        pd.testing.assert_frame_equal(df_with_bilateral_markers, original)


# ---------------------------------------------------------------------------
# Tests: integrate_dataframe_to_bird3D
# ---------------------------------------------------------------------------

class TestIntegrateDataframeToBird3D:
    @pytest.fixture
    def df_with_bird_markers(self):
        """DataFrame with all markers expected by integrate_dataframe_to_bird3D."""
        return pd.DataFrame({
            # Moving markers (left + right)
            'left_wingtip_x': [-0.3], 'left_wingtip_y': [0.1], 'left_wingtip_z': [0.0],
            'right_wingtip_x': [0.3], 'right_wingtip_y': [0.1], 'right_wingtip_z': [0.0],
            'left_primary_x': [-0.2], 'left_primary_y': [0.08], 'left_primary_z': [0.0],
            'right_primary_x': [0.2], 'right_primary_y': [0.08], 'right_primary_z': [0.0],
            'left_secondary_x': [-0.15], 'left_secondary_y': [0.05], 'left_secondary_z': [0.0],
            'right_secondary_x': [0.15], 'right_secondary_y': [0.05], 'right_secondary_z': [0.0],
            'left_tailtip_x': [-0.05], 'left_tailtip_y': [-0.1], 'left_tailtip_z': [0.0],
            'right_tailtip_x': [0.05], 'right_tailtip_y': [-0.1], 'right_tailtip_z': [0.0],
            # Fixed markers
            'left_shoulder_x': [-0.05], 'left_shoulder_y': [0.0], 'left_shoulder_z': [0.0],
            'right_shoulder_x': [0.05], 'right_shoulder_y': [0.0], 'right_shoulder_z': [0.0],
            'left_tailbase_x': [-0.02], 'left_tailbase_y': [-0.05], 'left_tailbase_z': [0.0],
            'right_tailbase_x': [0.02], 'right_tailbase_y': [-0.05], 'right_tailbase_z': [0.0],
            'hood_x': [0.0], 'hood_y': [0.05], 'hood_z': [0.0],
        })

    def test_returns_single_dict(self, df_with_bird_markers):
        markers = integrate_dataframe_to_bird3D(df_with_bird_markers)
        assert isinstance(markers, dict)

    def test_moving_markers_have_expected_keys(self, df_with_bird_markers):
        markers = integrate_dataframe_to_bird3D(df_with_bird_markers)
        expected_moving = {
            'left_wingtip', 'right_wingtip',
            'left_primary', 'right_primary',
            'left_secondary', 'right_secondary',
            'left_tailtip', 'right_tailtip',
        }
        assert expected_moving.issubset(markers.keys())

    def test_fixed_markers_have_expected_keys(self, df_with_bird_markers):
        markers = integrate_dataframe_to_bird3D(df_with_bird_markers)
        assert 'left_shoulder' in markers
        assert 'right_shoulder' in markers
        assert 'hood' in markers

    def test_marker_values_match_dataframe(self, df_with_bird_markers):
        markers = integrate_dataframe_to_bird3D(df_with_bird_markers)
        assert markers['left_wingtip'] == [
            pytest.approx(-0.3), pytest.approx(0.1), pytest.approx(0.0)]

    def test_hood_marker_values(self, df_with_bird_markers):
        markers = integrate_dataframe_to_bird3D(df_with_bird_markers)
        assert markers['hood'] == [pytest.approx(0.0), pytest.approx(0.05), pytest.approx(0.0)]


# ---------------------------------------------------------------------------
# Tests: fill_missing_body_measurements
# ---------------------------------------------------------------------------

class TestFillMissingBodyMeasurements:
    """Tests for wingspan-based fallback estimation of missing body measurements."""

    @pytest.fixture
    def df_complete(self):
        """DataFrame with all measurements present — nothing should change."""
        return pd.DataFrame({
            'wing_span_cm': [100.0],
            'body_width_max_cm': [15.0],
            'width_at_leg_insert_cm': [8.0],
            'tail_width_cm': [10.0],
            'tail_length_cm': [20.0],
            'head_length_cm': [9.0],
            'neck_length_cm': [10.0],
        })

    @pytest.fixture
    def df_with_nan_measurements(self):
        """DataFrame mimicking a species with NaN body measurements but known wingspan."""
        return pd.DataFrame({
            'wing_span_cm': [100.0],
            'body_width_max_cm': [np.nan],
            'width_at_leg_insert_cm': [np.nan],
            'tail_width_cm': [np.nan],
            'tail_length_cm': [np.nan],
            'head_length_cm': [np.nan],
            'neck_length_cm': [np.nan],
        })

    def test_fills_nan_from_wingspan(self, df_with_nan_measurements):
        result = fill_missing_body_measurements(df_with_nan_measurements)
        assert result['body_width_max_cm'].notna().all()
        assert result['width_at_leg_insert_cm'].notna().all()
        assert result['tail_width_cm'].notna().all()
        assert result['tail_length_cm'].notna().all()
        assert result['head_length_cm'].notna().all()
        assert result['neck_length_cm'].notna().all()

    def test_estimates_use_correct_proportions(self, df_with_nan_measurements):
        """Estimated values should equal wingspan * documented fraction."""
        from kinematic_morphospace.cross_species import _WINGSPAN_PROPORTIONS
        result = fill_missing_body_measurements(df_with_nan_measurements)
        wingspan = df_with_nan_measurements['wing_span_cm'].iloc[0]
        for col, fraction in _WINGSPAN_PROPORTIONS.items():
            if col in result.columns:
                assert result[col].iloc[0] == pytest.approx(wingspan * fraction), \
                    f"{col} estimate incorrect"

    def test_does_not_overwrite_existing_values(self, df_complete):
        result = fill_missing_body_measurements(df_complete)
        assert result['body_width_max_cm'].iloc[0] == pytest.approx(15.0)
        assert result['tail_length_cm'].iloc[0] == pytest.approx(20.0)

    def test_leaves_nan_when_wingspan_also_nan(self):
        """When wingspan is also missing, measurements cannot be estimated."""
        df = pd.DataFrame({
            'wing_span_cm': [np.nan],
            'body_width_max_cm': [np.nan],
            'tail_width_cm': [np.nan],
        })
        result = fill_missing_body_measurements(df)
        assert result['body_width_max_cm'].isna().all()
        assert result['tail_width_cm'].isna().all()

    def test_does_not_modify_original(self, df_with_nan_measurements):
        original = df_with_nan_measurements.copy()
        _ = fill_missing_body_measurements(df_with_nan_measurements)
        pd.testing.assert_frame_equal(df_with_nan_measurements, original)

    def test_mixed_missing_partial_rows(self):
        """Only NaN cells should be filled; present values preserved."""
        df = pd.DataFrame({
            'wing_span_cm': [80.0, 60.0],
            'body_width_max_cm': [12.0, np.nan],
            'tail_width_cm': [np.nan, 5.0],
        })
        result = fill_missing_body_measurements(df)
        # Row 0: body_width_max_cm unchanged, tail_width_cm filled
        assert result['body_width_max_cm'].iloc[0] == pytest.approx(12.0)
        assert result['tail_width_cm'].iloc[0] == pytest.approx(80.0 * 0.10)
        # Row 1: body_width_max_cm filled, tail_width_cm unchanged
        assert result['body_width_max_cm'].iloc[1] == pytest.approx(60.0 * 0.12)
        assert result['tail_width_cm'].iloc[1] == pytest.approx(5.0)

    def test_accepts_custom_proportions(self, df_with_nan_measurements):
        custom = {'body_width_max_cm': 0.25}
        result = fill_missing_body_measurements(df_with_nan_measurements, proportions=custom)
        assert result['body_width_max_cm'].iloc[0] == pytest.approx(25.0)
        # Other columns unchanged (still NaN since custom dict doesn't include them)
        assert result['tail_width_cm'].isna().all()


# ---------------------------------------------------------------------------
# Tests: missing measurements propagation through compute_derived_markers and
# check_and_fix_shoulder_distance (regression tests for PCA-paper-5uh)
# ---------------------------------------------------------------------------

class TestMissingMeasurementFallback:
    """Regression tests: species with NaN body measurements must not produce near-zero markers."""

    @pytest.fixture
    def base_wing_df(self):
        """Minimal wing DataFrame with all required pt markers."""
        return pd.DataFrame({
            'BirdID': ['bird_01'],
            'pt1_X': [0.01], 'pt1_Y': [0.0], 'pt1_Z': [0.0],
            'pt2_X': [0.05], 'pt2_Y': [0.0], 'pt2_Z': [0.01],
            'pt4_X': [0.1], 'pt4_Y': [0.05], 'pt4_Z': [0.0],
            'pt8_X': [0.2], 'pt8_Y': [0.1], 'pt8_Z': [0.0],
            'pt9_X': [0.3], 'pt9_Y': [0.15], 'pt9_Z': [0.0],
            'pt10_X': [0.15], 'pt10_Y': [0.08], 'pt10_Z': [0.0],
            'pt11_X': [0.0], 'pt11_Y': [-0.1], 'pt11_Z': [0.0],
            'pt12_X': [-0.2], 'pt12_Y': [0.1], 'pt12_Z': [0.0],
        })

    @pytest.fixture
    def df_nan_measurements(self, base_wing_df):
        """Wing DataFrame with NaN body measurements but valid wingspan — mirrors problem species."""
        df = base_wing_df.copy()
        df['wing_span_cm'] = 84.0     # realistic (e.g. peregrine falcon)
        df['body_width_max_cm'] = np.nan
        df['y_loc_of_humeral_insert_cm'] = np.nan
        df['width_at_leg_insert_cm'] = np.nan
        df['tail_width_cm'] = np.nan
        df['tail_length_cm'] = np.nan
        df['head_length_cm'] = np.nan
        df['neck_length_cm'] = np.nan
        df['species_common'] = ['peregrine_falcon']
        return df

    def test_compute_derived_markers_no_nan_output(self, df_nan_measurements):
        """compute_derived_markers must not produce NaN marker coordinates."""
        result = compute_derived_markers(df_nan_measurements)
        marker_cols = [c for c in result.columns if any(
            c.startswith(s) for s in ('left_', 'right_', 'hood_')
        )]
        for col in marker_cols:
            assert result[col].notna().all(), f"NaN found in column {col}"

    def test_tailtip_y_not_nan_with_missing_tail_length(self, df_nan_measurements):
        """tailtip_y is derived using tail_length_cm; must not be NaN after fill."""
        result = compute_derived_markers(df_nan_measurements)
        assert result['left_tailtip_y'].notna().all()
        assert result['right_tailtip_y'].notna().all()

    def test_hood_y_not_nan_with_missing_head_length(self, df_nan_measurements):
        """hood_y is derived using head_length_cm; must not be NaN after fill."""
        result = compute_derived_markers(df_nan_measurements)
        assert result['hood_y'].notna().all()

    def test_tailbase_not_near_zero_after_fix(self, df_nan_measurements):
        """Regression: tailbase x-width must be meaningfully non-zero after shoulder fix.

        Without the fallback, width_at_leg_insert_cm = NaN makes the mask False,
        the tailbase correction is skipped, and markers stay at near-zero x.
        """
        derived = compute_derived_markers(df_nan_measurements)

        # Build a minimal bilateral DataFrame for check_and_fix_shoulder_distance
        bilateral_cols = {
            'left_shoulder_x': derived['left_shoulder_x'],
            'left_shoulder_y': derived['left_shoulder_y'],
            'left_shoulder_z': derived['left_shoulder_z'],
            'right_shoulder_x': derived['right_shoulder_x'],
            'right_shoulder_y': derived['right_shoulder_y'],
            'right_shoulder_z': derived['right_shoulder_z'],
            'left_wingtip_x': derived['left_wingtip_x'],
            'left_wingtip_y': derived['left_wingtip_y'],
            'left_wingtip_z': derived['left_wingtip_z'],
            'right_wingtip_x': derived['right_wingtip_x'],
            'right_wingtip_y': derived['right_wingtip_y'],
            'right_wingtip_z': derived['right_wingtip_z'],
            'left_primary_x': derived['left_primary_x'],
            'left_primary_y': derived['left_primary_y'],
            'left_primary_z': derived['left_primary_z'],
            'right_primary_x': derived['right_primary_x'],
            'right_primary_y': derived['right_primary_y'],
            'right_primary_z': derived['right_primary_z'],
            'left_secondary_x': derived['left_secondary_x'],
            'left_secondary_y': derived['left_secondary_y'],
            'left_secondary_z': derived['left_secondary_z'],
            'right_secondary_x': derived['right_secondary_x'],
            'right_secondary_y': derived['right_secondary_y'],
            'right_secondary_z': derived['right_secondary_z'],
            'left_tailtip_x': derived['left_tailtip_x'],
            'left_tailtip_y': derived['left_tailtip_y'],
            'left_tailtip_z': derived['left_tailtip_z'],
            'right_tailtip_x': derived['right_tailtip_x'],
            'right_tailtip_y': derived['right_tailtip_y'],
            'right_tailtip_z': derived['right_tailtip_z'],
            'left_tailbase_x': derived['left_tailbase_x'],
            'left_tailbase_y': derived['left_tailbase_y'],
            'left_tailbase_z': derived['left_tailbase_z'],
            'right_tailbase_x': derived['right_tailbase_x'],
            'right_tailbase_y': derived['right_tailbase_y'],
            'right_tailbase_z': derived['right_tailbase_z'],
            'body_width_max_cm': df_nan_measurements['body_width_max_cm'],
            'y_loc_of_humeral_insert_cm': df_nan_measurements['y_loc_of_humeral_insert_cm'],
            'width_at_leg_insert_cm': df_nan_measurements['width_at_leg_insert_cm'],
            'tail_width_cm': df_nan_measurements['tail_width_cm'],
            'wing_span_cm': df_nan_measurements['wing_span_cm'],
            'pt1_X': df_nan_measurements['pt1_X'],
            'pt2_X': df_nan_measurements['pt2_X'],
        }
        bilateral_df = pd.DataFrame(bilateral_cols).reset_index(drop=True)

        result = check_and_fix_shoulder_distance(bilateral_df)

        tailbase_width = abs(
            result['right_tailbase_x'].iloc[0] - result['left_tailbase_x'].iloc[0]
        )
        # Expected tailbase width ≈ 2 * (wing_span_cm * 0.07) / 100
        # = 2 * (84 * 0.07) / 100 = 0.1176 m
        # We simply require it is not near-zero (> 0.01 m)
        assert tailbase_width > 0.01, (
            f"tailbase width {tailbase_width:.4f} m is near-zero — "
            "NaN measurement fallback may not be working"
        )
