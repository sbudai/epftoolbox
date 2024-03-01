"""
Simplified example for using the LEAR model for forecasting prices with daily recalibration
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

from epftoolbox.models import evaluate_lear_in_test_dataset
import os

# Market under study.
# If is not one of the standard ones, the file name has to be provided (without '.csv' extension),
# where the file has to be a csv file
dataset = 'DE'

# Name of the column in the input dataset that denotes the response_col variable.
response = 'Price'

# Number of years (a year is 364 days) in the test dataset.
years_test = 2

# Number of days used in the training dataset for recalibration
calibration_window = 364 * 3

# Define the path of the input dataset folder
path_datasets_folder = os.path.join('..', 'examples', 'datasets')

# Define the path to the folder where to save the forecasts
path_recalibration_folder = os.path.join('..', 'examples', 'experimental_files')
    
evaluate_lear_in_test_dataset(path_datasets_folder=path_datasets_folder,
                              path_recalibration_folder=path_recalibration_folder,
                              dataset=dataset,
                              response=response,
                              calibration_window=calibration_window,
                              years_test=years_test)
