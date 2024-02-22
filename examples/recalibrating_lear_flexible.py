"""
Example for using the LEAR model for forecasting prices with daily recalibration
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import pandas as pd
import numpy as np
import argparse
import os

from epftoolbox.data import read_and_split_data
from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.models import LEAR

# ------------------------------ EXTERNAL PARAMETERS ------------------------------------#

# initialize a parser object
parser = argparse.ArgumentParser()

# add requisite arguments to the parser
parser.add_argument("--dataset", type=str, default='PJM', 
                    help='Name of the file (w/o "csv" extension) containing the input dataset.')

parser.add_argument("--response", type=str, default='Zonal COMED price',
                    help=' Name of the column in the input dataset that denotes the response variable. '
                         'PJM - `Zonal COMED price` '
                         'NP - `Price` '
                         'BE - `Prices` '
                         'FR - `Prices` '
                         'DE - `Price`')

parser.add_argument("--calibration_window", type=int, default=3 * 364,
                    help='Number of days used in the training dataset for model recalibration.')

parser.add_argument("--years_test", type=int, default=2,
                    help=' Optional parameter to set the number of years of the being test dataset, '
                         'counting back from the end of the input dataset. '
                         'Note that a year is considered to be 364 days. '
                         'It is only used if the arguments `begin_test_date` and `end_test_date` are not provided.')

parser.add_argument("--begin_test_date", type=str, default=None, 
                    help='Optional parameter to select the test dataset. '
                         'Used in combination with the argument `end_test_date`. '
                         'If either of them is not provided, the test dataset will be split '
                         'using the `years_test` argument. The `begin_test_date` '
                         'should either be a string with the following format "%d/%m/%Y 00:00",'
                         'or a datetime object')

parser.add_argument("--end_test_date", type=str, default=None, 
                    help='Optional parameter to select the test dataset. '
                         'This value may be earlier than the last timestamp in the input dataset. '
                         'Used in combination with the argument `begin_test_date`. '
                         'If either of them is not provided, the test dataset will be split '
                         'using the `years_test` argument. The `end_test_date` '
                         'should either be a string with the following format "%d/%m/%Y 23:00",'
                         'or a datetime object.')

# parse the arguments and assign them to variables
args = parser.parse_args()
dataset = args.dataset
response = args.response
years_test = args.years_test
calibration_window = args.calibration_window
begin_test_date = args.begin_test_date
end_test_date = args.end_test_date

# Read the input data and split it to train and test set
df_train, df_test = read_and_split_data(path=os.path.join('.', 'examples', 'datasets'),
                                        dataset=dataset, response=response,
                                        years_test=years_test, begin_test_date=begin_test_date,
                                        end_test_date=end_test_date)

# Define unique name and path to save the forecasts
forecast_file_name = 'fc_nl_dat{0}_YT{1}_CW{2}.csv'.format(str(dataset), str(years_test), str(calibration_window))
forecast_file_path = os.path.join('.', 'examples', 'experimental_files', forecast_file_name)

# Extract real response (day-ahead price) values from the test set
# and pivot them wider by hour part of the timestamp index
real_values = df_test.loc[:, ['Price']]
real_values['column_hour'] = ['h' + h for h in real_values.index.strftime('%H').astype(int).astype(str)]
real_values = pd.pivot_table(data=real_values,
                             index=real_values.index.date,
                             columns='column_hour', aggfunc='mean', sort=False)
real_values.index = pd.to_datetime(real_values.index)

# Compose an empty forecast DataFrame with the same structure as the real values DataFrame has
forecast = real_values.copy()
forecast[:] = np.nan

# instantiate a LEAR model object
model = LEAR(calibration_window=calibration_window)

# Iterate over the model recalibration+forecast dates
for current_date in forecast.index:

    # For simulation purposes, we assume that the available data is
    # the data up to the end of recalibration+forecast date
    data_available = pd.concat(objs=[df_train, df_test.loc[:current_date + pd.Timedelta(hours=23), :]],
                               axis=0)

    # Set the real response variable values (day-ahead prices)
    # of the recalibration+forecast date to NaN in the dataframe of available data
    data_available.loc[current_date:current_date + pd.Timedelta(hours=23), response] = np.NaN

    # Recalibrate the model with the most up-to-date available data and make a prediction
    # for the recalibration+forecast day
    y_pred = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=current_date)

    # Save the current prediction into the forecast DataFrame
    forecast.loc[current_date, :] = y_pred

    # Compute metrics up to current recalibration+forecast date
    mae = np.mean(a=MAE(p_real=real_values.loc[:current_date].values,
                        p_pred=forecast.loc[:current_date].values.squeeze()))
    smape = np.mean(a=sMAPE(p_real=real_values.loc[:current_date].values,
                            p_pred=forecast.loc[:current_date].values.squeeze())) * 100

    # Print information
    print('{0} - sMAPE: {1:.2f}%  |  MAE: {2:.3f}'.format(str(current_date)[:10], smape, mae))

    # Save the forecast
    forecast.to_csv(path_or_buf=forecast_file_path)
