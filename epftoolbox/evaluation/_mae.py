"""
Function that implements the mean absolute error (MAE) metric.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from epftoolbox.evaluation._ancillary_functions import _process_inputs_for_metrics


def MAE(p_real, p_pred):
    """ Computes the mean absolute error (MAE) between two forecasts:

    .. math:: 
        \\mathrm{MAE} = \\frac{1}{N}\\sum_{i=1}^N \\bigl|p_\\mathrm{real}[i]-p_\\mathrm{pred}[i]\\bigr|    

    `p_real` and `p_pred` can either be of shape
    :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})`,
    :math:`(n_\\mathrm{prices}, 1)`, or :math:`(n_\\mathrm{prices}, )` where
    :math:`n_\\mathrm{prices} = n_\\mathrm{days} \\cdot n_\\mathrm{prices/day}`.

    Parameters
    ----------
    p_real : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the real prices.
    p_pred : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the predicted prices.
    
    Returns
    -------
    float
        The mean absolute error (MAE).

    Example
    --------

    >>> from epftoolbox.evaluation import MAE
    >>> from epftoolbox.data import read_and_split_data
    >>> import pandas as pd
    >>> 
    >>> # Download available day-ahead electricity price forecast of
    >>> # the Nord Pool market available in the library repository.
    >>> # These forecasts accompany the original paper
    >>> print('market: NP')
    >>> forecast = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/'
    ...                        'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)
    >>>
    >>> # Transforming the dataframe's timestamp indices to datetime format
    >>> forecast.index = pd.to_datetime(arg=forecast.index)
    >>> 
    >>> # Reading the real day-ahead electricity price data of the Nord Pool market
    >>> # The scope period should be same as in forecasted data.
    >>> _, df_test = read_and_split_data(path='.', dataset='NP', begin_test_date=forecast.index[0], 
    ...                                  end_test_date=forecast.index[-1])
    >>> # Test datasets: 2016-12-27 00:00:00 - 2018-12-24 23:00:00
    >>> 
    >>> # Extracting the day-ahead electricity price forecasts based on 'DNN Ensemble' model and display
    >>> fc_DNN_ensemble = forecast.loc[:, ['DNN Ensemble']]
    >>> print('fc_DNN_ensemble:', fc_DNN_ensemble, sep='\\n')
    >>> 
    >>> # Extracting the real day-ahead electricity price data and display
    >>> real_price = df_test.loc[:, ['Price']]
    >>> print('real_price:', real_price, sep='\\n')
    >>> 
    >>> # Building a 2-dimensional price forecast dataframe with shape (rows: n_days, columns: n_prices/n_day)
    >>> # instead of 1-dimensional shape (rows: n_prices, columns: 1) and display
    >>> fc_DNN_ensemble['column_hour'] = ['h' + h for h in fc_DNN_ensemble.index.strftime('%H').astype(int).astype(str)]
    >>> fc_DNN_ensemble_2D = pd.pivot_table(data=fc_DNN_ensemble, values='DNN Ensemble',
    ...                                     index=fc_DNN_ensemble.index.strftime('%Y-%m-%d'),
    ...                                     columns='column_hour', aggfunc='mean', sort=False)
    >>> fc_DNN_ensemble.drop(['column_hour'], axis='columns', inplace=True)
    >>> print('fc_DNN_ensemble_2D:', fc_DNN_ensemble_2D, sep='\\n')
    >>>
    >>> # Building a 2-dimensional real price dataframe with shape (rows: n_days, columns: n_prices/n_day)
    >>> # instead of 1-dimensional shape (rows: n_prices, columns: 1) and display
    >>> real_price['column_hour'] = ['h' + h for h in real_price.index.strftime('%H').astype(int).astype(str)]
    >>> real_price_2D = pd.pivot_table(data=real_price, values='Price',
    ...                                index=real_price.index.strftime('%Y-%m-%d'),
    ...                                columns='column_hour', aggfunc='mean', sort=False)
    >>> real_price.drop(['column_hour'], axis='columns', inplace=True)
    >>> print('real_price_2D:', real_price_2D, sep='\\n')
    >>>
    >>>
    >>> # According to the paper, the MAE of the 'DNN Ensemble' day-ahead price forecast for the NP market is 1.667
    >>> # Let's test the metric for different conditions
    >>> 
    >>> # Evaluating MAE when real day-ahead price and forecasts are both 1-dimensional dataframes
    >>> print('MAE(p_pred=fc_DNN_ensemble, p_real=real_price): {0}'.
    >>>       format(MAE(p_pred=fc_DNN_ensemble, p_real=real_price)))
    >>>
    >>> # Evaluating MAE when real day-ahead price and forecasts are both pandas Series
    >>> print('MAE(p_pred=fc_DNN_ensemble.loc[:, "DNN Ensemble"], p_real=real_price.loc[:, "Price"]): {0}'.
    >>>       format(MAE(p_pred=fc_DNN_ensemble.loc[:, 'DNN Ensemble'], p_real=real_price.loc[:, 'Price'])))
    >>>
    >>> # Evaluating MAE when real day-ahead price and forecasts are both 1-dimensional numpy arrays
    >>> print('MAE(p_pred=fc_DNN_ensemble.loc[:, "DNN Ensemble"].values,'
    >>>       ' p_real=real_price.loc[:, "Price"].values): {0}'.
    >>>       format(MAE(p_pred=fc_DNN_ensemble.loc[:, 'DNN Ensemble'].values,
    >>>                  p_real=real_price.loc[:, 'Price'].values)))
    >>>
    >>> # Evaluating MAE when real day-ahead price and forecasts are both 2-dimensional
    >>> # (rows: n_days, columns: n_prices/n_day)
    >>> # DataFrames
    >>> print('MAE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D): {0}'.
    >>>       format(MAE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D)))
    >>>
    >>> # Evaluating MAE when real day-ahead price and forecasts are both 2-dimensional
    >>> # (rows: n_days, columns: n_prices/n_day)
    >>> # numpy arrays
    >>> print('MAE(p_pred=fc_DNN_ensemble_2D.values.squeeze(), p_real=real_price_2D.values.squeeze()): {0}'.
    >>>       format(MAE(p_pred=fc_DNN_ensemble_2D.values.squeeze(), p_real=real_price_2D.values.squeeze())))
    >>> # 1.6670355192007669
    """

    # Checking if inputs are compatible
    p_real, p_pred = _process_inputs_for_metrics(p_real=p_real, p_pred=p_pred)

    return np.mean(np.abs(p_real - p_pred))
