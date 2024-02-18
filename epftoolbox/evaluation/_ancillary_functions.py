"""
Ancillary functions to compute accuracy metrics and statistical tests in the context of electricity price
forecasting
"""

import numpy as np
import pandas as pd


def _process_inputs_for_metrics(p_real: any, p_pred: any) -> tuple:
    """Function that checks that the two standard inputs of the metric functions satisfy some requirements
    
    
    Parameters
    ----------
    p_real : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the real prices
    p_pred : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the predicted prices
    
    Returns
    -------
    np.ndarray, np.ndarray
        The p_real and p_pred as numpy.ndarray objects after checking that they satisfy requirements 
    
    """

    # Checking whether both datasets are of the same allowed object type
    if (isinstance(p_real, pd.DataFrame) and not isinstance(p_pred, pd.DataFrame)) or\
            (isinstance(p_real, pd.Series) and not isinstance(p_pred, pd.Series)) or\
            (isinstance(p_real, np.ndarray) and not isinstance(p_pred, np.ndarray)):
        raise TypeError('The p_real and the p_pred objects must be of the same type. '
                        'The p_real is of type {0} and p_pred of type {1}'.format(type(p_real), type(p_pred)))

    # Checking whether datasets are of the allowed object types
    if not isinstance(p_real, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError('p_real and p_pred must be either a pandas.DataFrame, a pandas.Series, or a numpy.array. '
                        'They are of type {0}'.format(type(p_real)))

    # Transforming both datasets if they are pandas.Series to pandas.DataFrame
    if isinstance(p_real, pd.Series):
        p_real = p_real.to_frame()
        p_pred = p_pred.to_frame()
    
    # Checking whether datasets are pandas.DataFrames
    if isinstance(p_real, pd.DataFrame):
        # Checking whether both DataFrames share the same indices
        if not np.all((p_real.index == p_pred.index)):
            raise ValueError('p_real and p_pred must have the same indices')

        # Extracting their numeric values into numpy.ndarray
        p_real = p_real.select_dtypes(["number"]).values.squeeze()
        p_pred = p_pred.select_dtypes(["number"]).values.squeeze()

    return p_real, p_pred


def naive_forecast(p_real, m=None, n_prices_day=24):
    """ Function to build the naive forecast for electricity price forecasting.

    The seasonal naive forecasted values are equal to the last observed value from the same season.
    (The real observed value from the previous day or week.)
    
    The function is used to compute the accuracy metrics MASE and RMAE
        
    Parameters
    ----------
    p_real : pandas.DataFrame
        Dataframe containing the real prices. It must be of shape :math:`(n_\\mathrm{prices}, 1)`,
    m : int, optional
        Index that specifies the seasonality in the naive forecast. It can
        be ``'D'`` for daily seasonality, ``'W'`` for weekly seasonality, or ``None``
        for the standard naive forecast in electricity price forecasting, 
        i.e. daily seasonality for Tuesday to Friday and weekly seasonality 
        for Saturday to Monday.
    n_prices_day : int, optional
        Number of prices in a day. Usually this value is 24 for most day-ahead markets
    
    Returns
    -------
    pandas.DataFrame
        Dataframe containing the predictions of the naive forecast.
    """

    # Init the naive forecast
    if m is None or m == 'W':
        # remove the first 7 days from among the indices
        # index = p_real.index[n_prices_day * 7:]
        empty_frame = p_real[n_prices_day * 7:]
    else:
        # remove the first 1 day from among the indices
        # index = p_real.index[n_prices_day:]
        empty_frame = p_real[n_prices_day * 7:]

    # create an empty result DataFrame with the new indices
    # y_pred = pd.DataFrame(index=index, columns=p_real.columns)
    y_pred = pd.DataFrame().reindex_like(empty_frame)

    # If m is none the standard naive for EPF is built
    if m is None:

        # Monday we have a naive forecast using daily seasonality
        indices_mon = y_pred.index[y_pred.index.dayofweek == 0]
        y_pred.loc[indices_mon, :] = p_real.loc[indices_mon - pd.Timedelta(days=7), :].values.astype(float)

        # Tuesdays we have a naive forecast using daily seasonality
        indices_tue = y_pred.index[y_pred.index.dayofweek == 1]
        y_pred.loc[indices_tue, :] = p_real.loc[indices_tue - pd.Timedelta(days=1), :].values.astype(float)

        # Wednesday we have a naive forecast using daily seasonality
        indices_wed = y_pred.index[y_pred.index.dayofweek == 2]
        y_pred.loc[indices_wed, :] = p_real.loc[indices_wed - pd.Timedelta(days=1), :].values.astype(float)

        # Thursday we have a naive forecast using daily seasonality
        indices_thu = y_pred.index[y_pred.index.dayofweek == 3]
        y_pred.loc[indices_thu, :] = p_real.loc[indices_thu - pd.Timedelta(days=1), :].values.astype(float)

        # Friday we have a naive forecast using daily seasonality
        indices_fri = y_pred.index[y_pred.index.dayofweek == 4]
        y_pred.loc[indices_fri, :] = p_real.loc[indices_fri - pd.Timedelta(days=1), :].values.astype(float)

        # Saturday we have a naive forecast using weekly seasonality
        indices_sat = y_pred.index[y_pred.index.dayofweek == 5]
        y_pred.loc[indices_sat, :] = p_real.loc[indices_sat - pd.Timedelta(days=7), :].values.astype(float)

        # Sunday we have a naive forecast using weekly seasonality
        indices_sun = y_pred.index[y_pred.index.dayofweek == 6]
        y_pred.loc[indices_sun, :] = p_real.loc[indices_sun - pd.Timedelta(days=7), :].values.astype(float)

    # If m is either 'D' or 'W', then naive forecast simply built using a seasonal naive forecast
    # by filling up the empty result DataFrame with accordingly shifted values.
    elif m == 'D':
        y_pred.loc[:, :] = p_real.loc[y_pred.index - pd.Timedelta(days=1)].values.astype(float)
    elif m == 'W':
        y_pred.loc[:, :] = p_real.loc[y_pred.index - pd.Timedelta(days=7)].values.astype(float)

    return y_pred


def _transform_input_prices_for_naive_forecast(p_real, m, freq):
    """Function that ensures that the input of the naive forecast has the right format
    
    Parameters
    ----------
    p_real : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the real prices
    m : int, optional
        Index that specifies the seasonality in the naive forecast. It can
        be ``'D'`` for daily seasonality, ``'W'`` for weekly seasonality, or None
        for the standard naive forecast in electricity price forecasting, 
        i.e. daily seasonality for Tuesday to Friday and weekly seasonality 
        for Saturday to Monday.
    freq : str
        Frequency of the data if ``p_real`` are numpy.ndarray objects.
        It must take one of the following four values ``'1h'`` for 1 hour, ``'30min'`` for 30 minutes,
        ``'15min'`` for 15 minutes, or ``'5min'`` for 5 minutes,  (these are the four standard values in
        day-ahead electricity markets). If the shape of ``p_real`` is (n_days, n_prices_day),
        freq should be the frequency of the columns not the daily frequency of the rows.    
    Returns
    -------
    pandas.DataFrame
        ``p_real`` as a pandas.DataFrame that can be used for the naive forecast 
    """

    # Ensure that m value is correct
    if m not in ['D', 'W', None]: 
        raise ValueError('m argument has to be D, W, or None. Current values is {}'.format(m))

    # Check that input data is not numpy.ndarray and naive forecast is standard
    if m is None and not isinstance(p_real, pd.DataFrame) and not isinstance(p_real, pd.Series):
        raise TypeError('To use the standard naive forecast, i.e. m=None, the input '
                        'data has to be pandas.DataFrame object.')

    # Defining number of prices per day depending on frequency
    n_prices_day = {'1h': 24, '30min': 48, '15min': 96, '5min': 288, '1min': 1440}[freq]

    # If numpy arrays are used, ensure that there is integer number of days in the dataset
    if isinstance(p_real, np.ndarray) and p_real.size % n_prices_day != 0:
        raise ValueError('If numpy arrays are used, the size of p_real, i.e. the number of prices it '
                         'contains, should be a multiple number of {0}, i.e. of the number of '
                         'prices per day. Current values is {1}'.format(n_prices_day, p_real.size))
    
    # If pandas.Series are used, convert to DataFrame
    if isinstance(p_real, pd.Series):
        p_real = p_real.to_frame()

    # If input data is numpy.ndarray, transform to pandas.DataFrame
    if isinstance(p_real, np.ndarray):
        # Transforming p_real to correct shape, i.e. (n_prices, 1)
        p_real = p_real.reshape(-1, 1)
        # Building time indices for DataFrame
        indices = pd.date_range(start='2013-01-01', periods=p_real.shape[0], freq=freq)        
        # Building DataFrame
        p_real = pd.DataFrame(p_real, index=indices)
    
    # If input data is pandas-based, make sure it is in correct shape
    elif isinstance(p_real, pd.DataFrame):
        # Making sure that index is of datetime format
        p_real.index = pd.to_datetime(p_real.index)

        # Raising error if frequency cannot be inferred
        if p_real.index.inferred_freq is None:
            raise ValueError('The frequency/time periodicity of the data could not be inferred. '
                             'Ensure that the indices of the dataframe have a correct format '
                             'and are equally separated.')

        # If shape (n_days, n_prices_day), ensure that frequency of index is daily
        if p_real.shape[1] > 1 and p_real.index.inferred_freq != 'D':
            raise ValueError('If pandas dataframes are used with arrays with shape ' 
                             '(n_days, n_prices_day), the frequency of the time indices should be 1 day. '
                             'At the moment it is {0}.'.format(p_real.index.inferred_freq))

        # Reshaping dataframe if shape (n_days, n_prices_day)
        if p_real.shape[1] > 1:
            # Inferring frequency within a day
            frequency_seconds = 24 * 60 * 60 / p_real.shape[1]
            # Inferring last date in the dataset based on the frequency of points within a day
            last_date = p_real.index[-1] + (p_real.shape[1] - 1) * pd.Timedelta(seconds=frequency_seconds)
            # Inferring indices
            indices = pd.date_range(start=p_real.index[0], end=last_date, periods=p_real.size)
            # Reshaping prices
            p_real = pd.DataFrame(data=p_real.values.reshape(-1, 1), columns=['Prices'], index=indices)

    # Raising error if p_real not of specified type
    else:
        raise TypeError('Input should be of type numpy.ndarray, pandas.DataFrame, or pandas.Series '
                        ' but it is of type {0}'.format(type(p_real)))

    return p_real

