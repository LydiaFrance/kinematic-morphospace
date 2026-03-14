"""
test_data_loading.py

This module contains tests for the data loading and processing functions in the kinematic-morphospace project.
The tests ensure that the functions handle data correctly, remove invalid frames, process the data
into the correct format, and scale the data appropriately.

Tests include:
- Loading data from CSV files
- Removing invalid frames based on specific criteria
- Processing data into marker arrays and frame information
- Scaling marker data based on wingspan measurements

Each test is designed to verify specific aspects of the data loading and processing pipeline to ensure
robustness and correctness.

To run these tests, use the following command:
    pytest tests/
"""


import logging
import pytest
import pandas as pd
import numpy as np
import yaml

from kinematic_morphospace import load_data, remove_frames, process_data, filter_by
from kinematic_morphospace.data_scaling import scale_data

# Get the logger for this module
logger = logging.getLogger(__name__)

# -- Data Fixtures --

@pytest.fixture
def loaded_unilateral_data(sample_unilateraldata_path):
    """
    Load the sample unilateral data using the load_data function.
    """
    
    logger.info(f"Loading data using load_data function with {sample_unilateraldata_path}")
    return load_data(sample_unilateraldata_path)

@pytest.fixture
def loaded_bilateral_data(sample_bilateraldata_path):
    """
    Load the sample bilateral data using the load_data function.
    """
    logger.info(f"Loading data using load_data function with {sample_bilateraldata_path}")
    return load_data(sample_bilateraldata_path)


# -- Tests --   

# Initially testing loading data is working with path names
@pytest.mark.parametrize("data_fixture", ["sample_unilateraldata_path", "sample_bilateraldata_path"])
def test_load_data(request, data_fixture):
    """
    Test the load_data function with sample data.

    This test checks that the load_data function correctly loads a CSV file into a pandas DataFrame.
    It verifies that the DataFrame is not empty and that the data types are as expected.
    
    """
    logger.info(f"Testing load_data function with {data_fixture}")

    # Use the fixture sample_unilateraldata instead of loading the CSV directly
    data_path = request.getfixturevalue(data_fixture)
    data_csv = load_data(data_path)

    assert isinstance(data_csv, pd.DataFrame), "load_data should return a DataFrame"
    assert not data_csv.empty, "DataFrame should not be empty"
    logger.info("load_data function passed")


# Now testing with loaded data
@pytest.mark.parametrize("data_fixture", ["loaded_unilateral_data", "loaded_bilateral_data"])
def test_remove_frames(request, data_fixture):
    """
    Test the remove_frames function with sample data.
    
    This test checks that the remove_frames function correctly removes frames based on the 'time'
    and 'HorzDistance' criteria. It ensures that the remaining frames meet the specified conditions.
    """
    logger.info(f"Testing remove_frames function with {data_fixture}")

    # Get the fixture value based on the parameter
    # In other words, get the loaded data using the fixture name

    data_csv = request.getfixturevalue(data_fixture)
    initial_frame_count = len(data_csv)
    
    # Apply the remove_frames function
    cleaned_data = remove_frames(data_csv)

    # Ensure the cleaned data is a DataFrame
    assert isinstance(cleaned_data, pd.DataFrame), "remove_frames should return a DataFrame"
    assert len(cleaned_data) <= initial_frame_count, "remove_frames should not add frames"

    # Check that the remaining frames meet the criteria based on time and HorzDistance
    assert all(cleaned_data['time'] > 0), "All time values should be > time_limit after removing frames"

    logger.info("remove_frames function passed")


@pytest.mark.parametrize("data_fixture", ["loaded_unilateral_data", "loaded_bilateral_data"])
def test_process_data(request, data_fixture):
    """
    Test the process_data function with sample data.

    This test verifies that the process_data function correctly processes the data into marker arrays
    and frame information dictionaries. It ensures the output data structures are of the expected types
    and shapes.
    """
    logger.info(f"Testing process_data function with {data_fixture}")

    # Get the fixture value based on the parameter
    data_csv = request.getfixturevalue(data_fixture)
    
    # Process the data
    markers, frame_info, markers_df, frame_info_df = process_data(data_csv)

    # Verify the output types and shapes
    assert isinstance(markers, np.ndarray), "markers should be a numpy array"
    assert isinstance(frame_info, dict), "frame_info should be a dictionary"
    assert isinstance(markers_df, pd.DataFrame), "markers_df should be a DataFrame"
    assert isinstance(frame_info_df, pd.DataFrame), "frame_info_df should be a DataFrame"
    assert markers.shape[0] == frame_info['time'].shape[0], "Markers and frame info should have the same number of frames"
    assert markers_df.shape[0] == frame_info_df.shape[0], "DataFrames should have the same number of rows"
    logger.info("process_data function passed")


@pytest.mark.parametrize("data_fixture", ["loaded_unilateral_data", "loaded_bilateral_data"])
def test_scale_data(request, data_fixture, sample_wingspan_path):
    """
    Test the scale_data function with sample data.

    This test checks that the scale_data function scales marker data correctly based on the wingspan
    measurements. It ensures that no NaN values are introduced and that the data remains valid.
    """
    
    logger.info(f"Testing scale_data function with {data_fixture}")

    # Loads the ficture data based on the parameter
    data_csv = request.getfixturevalue(data_fixture)

    # Apply the scale_data function, and uses the test wingspan yaml file
    scaled_data = scale_data(data_csv, sample_wingspan_path)

    # Verify the output types and integrity
    assert isinstance(scaled_data, pd.DataFrame), "scale_data should return a DataFrame"
    marker_cols = [col for col in scaled_data.columns if '_x' in col or '_y' in col or '_z' in col]
    assert not scaled_data[marker_cols].isnull().values.any(), "Scaled marker data should not contain NaNs"
    logger.info("scale_data function passed")


@pytest.mark.parametrize("data_fixture", ["loaded_unilateral_data", "loaded_bilateral_data"])
def test_data_integrity(request, data_fixture):
    """
    Test the integrity of the data after processing.

    This test checks that the data has the same columns and data types after removing frames.
    """
    logger.info(f"Testing data integrity for {data_fixture}")

    data_csv = request.getfixturevalue(data_fixture)
    original_columns = data_csv.columns
    original_dtypes = data_csv.dtypes

    cleaned_data = remove_frames(data_csv)
    assert list(cleaned_data.columns) == list(original_columns), "Column names should remain the same after removing frames"
    assert (cleaned_data.dtypes == original_dtypes).all(), "Data types should remain the same after removing frames"

    logger.info("Data integrity test passed")

@pytest.mark.parametrize("data_fixture", ["loaded_unilateral_data", "loaded_bilateral_data"])
def test_scale_data_accuracy(request, data_fixture, sample_wingspan_path):
    """
    Test the accuracy of the scale_data function.

    This test checks that the scale_data function scales the marker data correctly based on the wingspan
    """
    
    logger.info(f"Testing scale_data function accuracy with {data_fixture}")

    # Loads the fixture data based on the parameter
    data_csv = request.getfixturevalue(data_fixture)

    # Apply the scale_data function, and uses the test wingspan yaml file
    scaled_data = scale_data(data_csv, sample_wingspan_path)

    assert isinstance(scaled_data, pd.DataFrame), "scale_data should return a DataFrame"
    
    # Load yml file as dictionary to check directly
    with open(sample_wingspan_path) as file:
        total_wingspans = yaml.load(file, Loader=yaml.FullLoader)
    
    # Check if the data has been scaled correctly
    marker_cols = [col for col in scaled_data.columns if '_x' in col or '_y' in col or '_z' in col]
    for hawk in ["Drogon", "Rhaegal", "Ruby", "Toothless", "Charmander"]:
        for year in [2017, 2020]:
            filter = filter_by(scaled_data, hawkname=hawk, year=year)
            
            # Skip if no data for the hawk and year
            if sum(filter) == 0:
                continue

            # Get the total wingspan for the hawk
            wingspan = total_wingspans[hawk][year]
            
            # Check that the scaled data does not exceed the wingspan values
            if sum(filter) > 0:
                assert (scaled_data.loc[filter, marker_cols] <= wingspan).all().all(), "Scaled data should not exceed wingspan values"
    
    logger.info("scale_data function accuracy test passed")
