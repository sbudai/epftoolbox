from epftoolbox.evaluation import plot_multivariate_DM_test
from epftoolbox.data import read_and_split_data
import pandas as pd
import os

# Generate forecasts of multiple models

# Download available forecast of the NP market available in the library repository
# These forecasts accompany the original paper
forecasts = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/'
                        'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)

# Deleting the real price field as it the actual real price and not a forecast
forecasts.drop(columns=['Real price'])

# Transforming indices to datetime format
forecasts.index = pd.to_datetime(arg=forecasts.index)

# Read the real day-ahead electricity price data of the Nord Pool market
# The scope period should be the same as in forecasted data.
_, df_test = read_and_split_data(path=os.path.join('..', '..', 'examples', 'datasets'),
                                 dataset='NP', response='Price',
                                 begin_test_date=forecasts.index[0], end_test_date=forecasts.index[-1])

# Extract the real day-ahead electricity price data and display
real_price = df_test.loc[:, ['Price']]

# Plot the comparison of the models using the multivariate DM test
plot_multivariate_DM_test(real_price=real_price, forecasts=forecasts)
