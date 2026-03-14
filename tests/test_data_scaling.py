"""
test_data_scaling.py

Tests for the data_scaling module: scale_data(), unscale_data(),
add_turn_info(), add_tailpack_data(), rename_tailpack_data().
Exposes known issue #3 (hardcoded default path).
"""

import logging
import pytest
import numpy as np
import pandas as pd
import yaml

from kinematic_morphospace import load_data, process_data, filter_by
from kinematic_morphospace.data_scaling import (
    scale_data,
    unscale_data,
    rename_tailpack_data,
)

logger = logging.getLogger(__name__)


# -- Fixtures --

@pytest.fixture
def loaded_unilateral(sample_unilateraldata_path):
    return load_data(sample_unilateraldata_path)


@pytest.fixture
def loaded_bilateral(sample_bilateraldata_path):
    return load_data(sample_bilateraldata_path)


# -- scale_data tests --

class TestScaleData:
    def test_scale_returns_dataframe(self, loaded_unilateral, sample_wingspan_path):
        scaled = scale_data(loaded_unilateral.copy(), sample_wingspan_path)
        assert isinstance(scaled, pd.DataFrame)

    def test_scale_no_nans(self, loaded_unilateral, sample_wingspan_path):
        scaled = scale_data(loaded_unilateral.copy(), sample_wingspan_path)
        marker_cols = [c for c in scaled.columns if '_x' in c or '_y' in c or '_z' in c]
        assert not scaled[marker_cols].isnull().values.any()

    def test_scaling_reduces_values(self, loaded_unilateral, sample_wingspan_path):
        """Scaled marker values should be smaller than originals (divided by wingspan > 1)."""
        original = loaded_unilateral.copy()
        scaled = scale_data(loaded_unilateral.copy(), sample_wingspan_path)
        marker_cols = [c for c in scaled.columns if '_x' in c or '_y' in c or '_z' in c]
        # Absolute values should be smaller after dividing by wingspan > 1
        orig_abs_mean = original[marker_cols].abs().mean().mean()
        scaled_abs_mean = scaled[marker_cols].abs().mean().mean()
        assert scaled_abs_mean < orig_abs_mean

    def test_wingspan_path_is_required(self, loaded_unilateral):
        """Fixed issue #3: wingspan_path no longer has a hardcoded default."""
        with pytest.raises(TypeError):
            scale_data(loaded_unilateral.copy())


# -- unscale_data tests --

class TestUnscaleData:
    def test_scale_unscale_roundtrip(self, loaded_unilateral, sample_wingspan_path):
        """Scaling then unscaling should recover original values."""
        original = loaded_unilateral.copy()
        marker_cols = [c for c in original.columns if '_x' in c or '_y' in c or '_z' in c]

        scaled = scale_data(original.copy(), sample_wingspan_path)
        restored = unscale_data(scaled.copy(), sample_wingspan_path)

        np.testing.assert_allclose(
            restored[marker_cols].values,
            original[marker_cols].values,
            rtol=1e-10,
            err_msg="Scale/unscale roundtrip should recover original values"
        )

    def test_unscale_on_bilateral(self, loaded_bilateral, sample_wingspan_path):
        """Roundtrip should also work on bilateral data."""
        original = loaded_bilateral.copy()
        marker_cols = [c for c in original.columns if '_x' in c or '_y' in c or '_z' in c]

        scaled = scale_data(original.copy(), sample_wingspan_path)
        restored = unscale_data(scaled.copy(), sample_wingspan_path)

        np.testing.assert_allclose(
            restored[marker_cols].values,
            original[marker_cols].values,
            rtol=1e-10,
        )


# -- rename_tailpack_data tests --

class TestRenameTailpackData:
    def test_renames_columns(self):
        df = pd.DataFrame({
            'frameID': ['f1', 'f2'],
            'rot_xyz_1': [0.1, 0.2],
            'rot_xyz_2': [0.3, 0.4],
            'rot_xyz_3': [0.5, 0.6],
            'xyz_1': [1.0, 2.0],
            'xyz_2': [3.0, 4.0],
            'xyz_3': [5.0, 6.0],
        })
        result = rename_tailpack_data(df)
        assert 'tailpack_x' in result.columns
        assert 'tailpack_y' in result.columns
        assert 'tailpack_z' in result.columns
        # Original xyz columns should be dropped
        assert 'xyz_1' not in result.columns
        assert 'xyz_2' not in result.columns
        assert 'xyz_3' not in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            'rot_xyz_1': [0.1],
            'rot_xyz_2': [0.3],
            'rot_xyz_3': [0.5],
            'xyz_1': [1.0],
            'xyz_2': [3.0],
            'xyz_3': [5.0],
        })
        _ = rename_tailpack_data(df)
        # Original should still have old column names
        assert 'rot_xyz_1' in df.columns
