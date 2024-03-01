"""
Function that implements the root-mean-square error (RMSE) metric.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from epftoolbox.evaluation._ancillary_functions import _process_inputs_for_metrics


def RMSE(p_real, p_pred):
    """ Computes the root mean square error (RMSE) between predicted and observed values:

    .. math:: \\mathrm{RMSE} = \\frac{1}{N}\\sum_{i=1}^N \\sqrt{\\bigl(p_\\mathrm{real}[i]−p_\\mathrm{pred}[i]\\bigr)^2}    
    
    ``p_real`` and ``p_pred`` can either be of shape 
    :math:`(n_\\mathrm{days}, n_\\mathrm{prices/day})`,
    :math:`(n_\\mathrm{prices}, 1)`, or :math:`(n_\\mathrm{prices}, )` where
    :math:`n_\\mathrm{prices} = n_\\mathrm{days} \\cdot n_\\mathrm{prices/day}`.

    Parameters
    ----------
        p_real : numpy.ndarray | pandas.DataFrame
            Array/dataframe containing the observed prices.
        p_pred : numpy.ndarray | pandas.DataFrame
            Array/dataframe containing the predicted prices.
    
    Returns
    -------
        float
            The root mean square error (RMSE) value.
    """
    
    # Check if inputs are compatible
    p_real, p_pred = _process_inputs_for_metrics(p_real=p_real, p_pred=p_pred)
    
    # Calculate the squared error at every time point
    se = (p_real - p_pred) ** 2    

    return np.sqrt(np.mean(a=se))


if __name__ == '__main__':
    # These codes below will only be executed if this script is run as the main program.
    from epftoolbox.data import read_and_split_data
    import pandas as pd
    import os

    # Download available day-ahead electricity price forecast of
    # the Nord Pool market available in the library repository.
    # These forecasts accompany the original paper
    print('market: NP')
    forecast = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/'
                           'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)

    # Extract the forecasted day-ahead electricity prices based on 'DNN Ensemble' model and display
    fc_DNN_ensemble = forecast.loc[:, ['DNN Ensemble']]
    print('fc_DNN_ensemble:', fc_DNN_ensemble, sep='\n')
    # fc_DNN_ensemble:
    #                      DNN Ensemble
    # 2016-12-27 00:00:00     24.901055
    # 2016-12-27 01:00:00     23.496779
    # 2016-12-27 02:00:00     22.808439
    # 2016-12-27 03:00:00     22.529546
    # 2016-12-27 04:00:00     23.217815
    # ...
    # 2018-12-24 19:00:00     50.993732
    # 2018-12-24 20:00:00     49.430166
    # 2018-12-24 21:00:00     48.648760
    # 2018-12-24 22:00:00     47.358303
    # 2018-12-24 23:00:00     46.116690
    # [17472 rows x 1 columns]

    # Transform its indices to datetime format
    fc_DNN_ensemble.index = pd.to_datetime(arg=fc_DNN_ensemble.index)

    # Read the real day-ahead electricity price data of the Nord Pool market
    # The scope period should be the same as in forecasted data.
    _, df_test = read_and_split_data(path=os.path.join('..', '..', 'examples', 'datasets'),
                                     dataset='NP', response_col='Price',
                                     begin_test_date=fc_DNN_ensemble.index[0],
                                     end_test_date=fc_DNN_ensemble.index[-1])
    # Training dataset period: 2013-01-01 00:00:00 - 2016-12-26 23:00:00
    # Testing dataset period: 2016-12-27 00:00:00 - 2018-12-24 23:00:00

    # Extract the real day-ahead electricity price data and display
    real_price = df_test.loc[:, ['Price']]
    print('real_price:', real_price, sep='\n')
    # real_price:
    #                      Price
    # Date
    # 2016-12-27 00:00:00  24.08
    # 2016-12-27 01:00:00  22.52
    # 2016-12-27 02:00:00  20.13
    # 2016-12-27 03:00:00  19.86
    # 2016-12-27 04:00:00  20.09
    # ...
    # 2018-12-24 19:00:00  50.72
    # 2018-12-24 20:00:00  49.86
    # 2018-12-24 21:00:00  49.09
    # 2018-12-24 22:00:00  49.02
    # 2018-12-24 23:00:00  48.10
    # [17472 rows x 1 columns]

    # Build a 2-dimensional price forecast dataframe with shape (rows: n_days, columns: n_prices/n_day)
    # instead of 1-dimensional shape (rows: n_prices, columns: 1) and display
    fc_DNN_ensemble['column_hour'] = ['h' + h for h in fc_DNN_ensemble.index.strftime('%H').astype(int).astype(str)]
    fc_DNN_ensemble_2D = pd.pivot_table(data=fc_DNN_ensemble, values='DNN Ensemble',
                                        index=fc_DNN_ensemble.index.date,
                                        columns='column_hour', aggfunc='mean', sort=False)
    fc_DNN_ensemble_2D.index = pd.to_datetime(fc_DNN_ensemble_2D.index)
    fc_DNN_ensemble_2D.index.name = 'date'
    fc_DNN_ensemble_2D.columns.name = None
    fc_DNN_ensemble.drop(['column_hour'], axis='columns', inplace=True)
    print('fc_DNN_ensemble_2D:', fc_DNN_ensemble_2D, sep='\n')
    # fc_DNN_ensemble_2D:
    #                     h0         h1         h2  ...        h21        h22        h23
    # date                                         ...
    # 2016-12-27   24.901055  23.496779  22.808439  ...  28.640429  27.806076  26.643023
    # 2016-12-28   25.164850  24.544741  24.135444  ...  29.153526  28.375926  27.324923
    # 2016-12-29   28.330177  27.697056  27.186900  ...  28.204550  27.970621  27.199672
    # 2016-12-30   27.948486  27.374095  27.076586  ...  29.083482  28.470059  27.713009
    # 2016-12-31   26.366282  25.459522  24.848794  ...  27.410532  26.989756  26.243367
    # ...        ...        ...  ...        ...        ...        ...
    # 2018-12-20   50.030031  49.559196  48.983173  ...  52.382862  51.247468  50.062469
    # 2018-12-21   48.193556  47.452021  46.931173  ...  51.885920  50.595848  49.077818
    # 2018-12-22   47.561112  46.781378  46.285856  ...  50.142850  49.016890  47.803343
    # 2018-12-23   49.353018  48.732417  48.444989  ...  52.442492  50.556766  49.600979
    # 2018-12-24   49.099671  47.957044  47.300990  ...  48.648760  47.358303  46.116690
    # [728 rows x 24 columns]

    # Build a 2-dimensional real price dataframe with shape (rows: n_days, columns: n_prices/n_day)
    # instead of 1-dimensional shape (rows: n_prices, columns: 1) and display
    real_price['column_hour'] = ['h' + h for h in real_price.index.strftime('%H').astype(int).astype(str)]
    real_price_2D = pd.pivot_table(data=real_price, values='Price',
                                   index=real_price.index.date,
                                   columns='column_hour', aggfunc='mean', sort=False)
    real_price_2D.index = pd.to_datetime(real_price_2D.index)
    real_price_2D.index.name = 'date'
    real_price_2D.columns.name = None
    real_price.drop(['column_hour'], axis='columns', inplace=True)
    print('real_price_2D:', real_price_2D, sep='\n')
    # real_price_2D:
    #                 h0     h1     h2     h3 ...     h20    h21    h22    h23
    # date
    # 2016-12-27   24.08  22.52  20.13  19.86  ...  29.14  28.37  27.24  25.73
    # 2016-12-28   26.45  26.26  26.24  26.43  ...  31.24  30.65  30.02  29.37
    # 2016-12-29   29.26  28.72  28.29  28.32  ...  30.28  30.01  29.44  28.76
    # 2016-12-30   28.18  27.03  26.47  26.47  ...  29.56  28.96  27.77  25.95
    # 2016-12-31   25.11  23.65  22.99  22.13  ...  27.98  27.52  26.80  26.71
    # ...    ...    ...    ...  ...    ...    ...    ...    ...
    # 2018-12-20   48.35  48.08  48.31  47.36  ...  52.25  51.29  50.00  48.12
    # 2018-12-21   47.21  46.42  46.00  46.12  ...  52.05  51.43  50.07  49.01
    # 2018-12-22   48.39  47.72  47.23  46.60  ...  53.05  52.05  51.09  50.47
    # 2018-12-23   51.49  50.83  50.74  50.14  ...  55.61  53.99  53.86  52.32
    # 2018-12-24   51.09  50.19  48.98  48.80  ...  49.86  49.09  49.02  48.10
    # [728 rows x 24 columns]
    
    # According to the paper, the RMSE of the DNN ensemble for the NP market is 3.333.
    # Let's test the metric for different conditions

    # Evaluate the RMSE when real day-ahead prices and forecast are both 1-dimensional (long) dataframes
    print('RMSE(p_pred=fc_DNN_ensemble, p_real=real_price): {0}'.
          format(RMSE(p_pred=fc_DNN_ensemble, p_real=real_price)))
    # RMSE(p_pred=fc_DNN_ensemble, p_real=real_price): 3.3189883679864396

    # Evaluate the RMSE when real day-ahead price and forecasts are both pandas series
    print('RMSE(p_pred=fc_DNN_ensemble.loc[:, "DNN Ensemble"], p_real=real_price.loc[:, "Price"]): {0}'.
          format(RMSE(p_pred=fc_DNN_ensemble.loc[:, 'DNN Ensemble'], p_real=real_price.loc[:, 'Price'])))
    # RMSE(p_pred=fc_DNN_ensemble.loc[:, "DNN Ensemble"], p_real=real_price.loc[:, "Price"]): 3.3189883679864396

    # Evaluate the RMSE when real day-ahead price and forecasts are both 1-dimensional (long) numpy arrays
    print('RMSE(p_pred=fc_DNN_ensemble.loc[:, "DNN Ensemble"].values, p_real=real_price.loc[:, "Price"].values): {0}'.
          format(RMSE(p_pred=fc_DNN_ensemble.loc[:, 'DNN Ensemble'].values,
                      p_real=real_price.loc[:, 'Price'].values)))
    # RMSE(p_pred=fc_DNN_ensemble.loc[:, "DNN Ensemble"].values,
    #     p_real=real_price.loc[:, "Price"].values): 3.3189883679864396

    # Evaluate the RMSE when real day-ahead price and forecasts are both 2-dimensional (wide) DataFrames
    # (rows: n_days, columns: n_prices/n_day)
    print('RMSE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D): {0}'.
          format(RMSE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D)))
    # RMSE(p_pred=fc_DNN_ensemble_2D, p_real=real_price_2D): 3.31898836798644

    # Evaluate the RMSE when real day-ahead price and forecasts are both 2-dimensional (wide) numpy arrays
    # (rows: n_days, columns: n_prices/n_day)
    print('RMSE(p_pred=fc_DNN_ensemble_2D.values.squeeze(), p_real=real_price_2D.values.squeeze()): {0}'.
          format(RMSE(p_pred=fc_DNN_ensemble_2D.values.squeeze(), p_real=real_price_2D.values.squeeze())))
    # RMSE(p_pred=fc_DNN_ensemble_2D.values.squeeze(), p_real=real_price_2D.values.squeeze()): 3.31898836798644
