import sys
import os
import pytest
import logging
import coloredlogs
import pandas as pd

# Cap loky worker count so tests don't hit warnings on machines with
# many cores (see joblib / sklearn parallel backends).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Configure coloredlogs for all tests
coloredlogs.install(level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Reduce verbosity of matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# Define fixtures that can be used across multiple test files
@pytest.fixture
def sample_unilateraldata_path():
    data_path = os.path.join(os.path.dirname(__file__), 
                             'test_data', 
                             '2024-06-03-TestUnilateralMarkers.csv')
    return data_path


@pytest.fixture
def sample_bilateraldata_path():
    data_path = os.path.join(os.path.dirname(__file__), 
                             'test_data', 
                             '2024-06-03-TestBilateralMarkers.csv')
    return data_path

@pytest.fixture
def sample_wingspan_path():
    data_path = os.path.join(os.path.dirname(__file__), 
                             'test_data', 
                             'TotalWingspans.yml')
    return data_path