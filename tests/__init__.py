import logging
import os
import coloredlogs

# Create a logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Basic logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_results.log'),
        logging.StreamHandler()
    ]
)

# Configure coloredlogs
coloredlogs.install(
    level='DEBUG', 
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log a message indicating the start of the test suite
logging.info("Starting test suite")

# Optionally, you can also define some common fixtures or setup code here
os.environ['ENV'] = 'test'
