from epftoolbox.evaluation import MAPE
from epftoolbox.data import read_and_split_data
import pandas as pd

# Download available day-ahead electricity price forecasts of
# the Nord Pool market available in the library repository.
# These forecasts accompany the original paper.
print('market: Nord Pool')
forecast = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/'
                       'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)

# Transforming the dataframe's timestamp indices to datetime format
forecast.index = pd.to_datetime(arg=forecast.index)

# Reading the real day-ahead electricity price data of the Nord Pool market.
# The scope period should be the same as in forecasted data.
_, df_test = read_and_split_data(path='../datasets', dataset='NP', response_col='Price',
                                 begin_test_date=forecast.index[0], end_test_date=forecast.index[-1])

# Extracting the day-ahead electricity price forecasts based on 'DNN Ensemble' model and display
fc_DNN_ensemble = forecast.loc[:, ['DNN Ensemble']]
print('fc_DNN_ensemble:', fc_DNN_ensemble, sep='\n')

# Extracting the real day-ahead electricity price data and display
real_price = df_test.loc[:, ['Price']]
print('real_price:', real_price, sep='\n')

# Building a 2-dimensional price forecast dataframe with shape (rows: n_days, columns: n_prices/n_day)
# instead of 1-dimensional shape (rows: n_prices, columns: 1) and display
# The doubled (autumn) daylight saving time hour values have averaged out.
fc_DNN_ensemble['column_hour'] = ['h' + h for h in fc_DNN_ensemble.index.strftime('%H').astype(int).astype(str)]
fc_DNN_ensemble_2D = pd.pivot_table(data=fc_DNN_ensemble, values='DNN Ensemble',
                                    index=fc_DNN_ensemble.index.strftime('%Y-%m-%d'),
                                    columns='column_hour', aggfunc='mean', sort=False)
fc_DNN_ensemble.drop(['column_hour'], axis='columns', inplace=True)
fc_DNN_ensemble_2D.columns.name = None
print('fc_DNN_ensemble_2D:', fc_DNN_ensemble_2D, sep='\n')

# Building a 2-dimensional real price dataframe with shape (rows: n_days, columns: n_prices/n_day)
# instead of 1-dimensional shape (rows: n_prices, columns: 1) and display
# The doubled (autumn) daylight saving time hour values have averaged out.
real_price['column_hour'] = ['h' + h for h in real_price.index.strftime('%H').astype(int).astype(str)]
real_price_2D = pd.pivot_table(data=real_price, values='Price',
                               index=real_price.index.strftime('%Y-%m-%d'),
                               columns='column_hour', aggfunc='mean', sort=False)
real_price.drop(['column_hour'], axis='columns', inplace=True)
real_price_2D.columns.name = None
print('real_price_2D:', real_price_2D, sep='\n')


# According to the paper, the MAPE of the 'DNN Ensemble' day-ahead price forecast for the NP market is 5.38%.
# Let's test the metric for different conditions

# Evaluating MAPE when real day-ahead price and forecasts are both 1-dimensional dataframes
print('MAPE(p_real=real_price, p_pred=fc_DNN_ensemble): {0:5.2f}%'.
      format(MAPE(p_real=real_price, p_pred=fc_DNN_ensemble) * 100))

# Evaluating MAPE when real day-ahead price and forecasts are both pandas Series
print('MAPE(p_real=real_price.loc[:, "Price"], p_pred=fc_DNN_ensemble.loc[:, "DNN Ensemble"]): {0:5.2f}%'.
      format(MAPE(p_real=real_price.loc[:, 'Price'], p_pred=fc_DNN_ensemble.loc[:, 'DNN Ensemble']) * 100))

# Evaluating MAPE when real day-ahead price and forecasts are both 1-dimensional numpy arrays
print('MAPE(p_real=real_price.loc[:, "Price"].values, p_pred=fc_DNN_ensemble.loc[:, "DNN Ensemble"].values): {0:5.2f}%'.
      format(MAPE(p_real=real_price.loc[:, 'Price'].values,
                  p_pred=fc_DNN_ensemble.loc[:, 'DNN Ensemble'].values) * 100))

# Evaluating MAPE when real day-ahead price and forecasts are both 2-dimensional (rows: n_days, columns: n_prices/n_day)
# DataFrames
print('MAPE(p_real=real_price_2D, p_pred=fc_DNN_ensemble_2D): {0:5.2f}%'.
      format(MAPE(p_real=real_price_2D, p_pred=fc_DNN_ensemble_2D) * 100))

# Evaluating MAPE when real day-ahead price and forecasts are both 2-dimensional (rows: n_days, columns: n_prices/n_day)
# numpy arrays
print('MAPE(p_real=real_price_2D.values.squeeze(), p_pred=fc_DNN_ensemble_2D.values.squeeze()): {0:5.2f}%'.
      format(MAPE(p_real=real_price_2D.values.squeeze(), p_pred=fc_DNN_ensemble_2D.values.squeeze()) * 100))


# We can also test situations where the MAPE will display errors

# Evaluating MAPE when real day-ahead price and forecasts are of different object type
# (numpy.ndarray and pandas.DataFrame)
try:
    print('MAPE: {0:5.2f}%'.format(MAPE(p_real=real_price, p_pred=fc_DNN_ensemble.values) * 100))
except TypeError as e:
    print("TypeError:", e)

# Evaluating MAPE when real day-ahead price and forecasts are of the different object type
# (pandas.Series and pandas.DataFrame)
try:
    print('MAPE: {0:5.2f}%'.format(MAPE(p_real=real_price.loc[:, 'Price'], p_pred=fc_DNN_ensemble) * 100))
except TypeError as e:
    print("TypeError:", e)

# Evaluating MAPE when real day-ahead price and forecasts are both numpy arrays but of different size
try:
    print('MAPE: {0:5.2f}%'.format(MAPE(p_real=real_price.values[1:], p_pred=fc_DNN_ensemble.values) * 100))
except ValueError as e:
    print("ValueError:", e)

# Evaluating MAPE when real day-ahead price and forecasts are both DataFrames but of different size
try:
    print('MAPE: {0:5.2f}%'.format(MAPE(p_real=real_price, p_pred=fc_DNN_ensemble.iloc[1:, :]) * 100))
except ValueError as e:
    print("ValueError:", e)
