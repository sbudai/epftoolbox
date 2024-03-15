"""
Classes and functions to implement the LEAR model for day-ahead electricity price forecasting
"""

# Author: Jesus Lago & Sandor Budai

# License: AGPL-3.0 License

import numpy as np
import pandas as pd
import os
from datetime import datetime
from epftoolbox.data import read_data, split_data, scaling
from epftoolbox.evaluation import MAE, sMAPE
from sklearn.linear_model import LassoLarsIC, Lasso
# noinspection PyProtectedMember
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class LEAR(object):
    """ Class to build a LEAR model, recalibrate it, and use it to predict day-ahead electricity prices. """

    def __init__(self, calibration_window=364 * 3, normalize='Invariant', criterion='aic', max_iter=2500,
                 price_lags=(1, 2, 3, 7), exog_lags=(0, 1, 2), dow_dummies=True, daily_delivery_period_numbers=None):
        """ Instantiate a LEAR model object.

        Parameters
        ----------
            calibration_window : int
                The number of days (a year is considered as 364 days)
                that are used in the training dataset for recalibration.
                Limits the training dataset starting point in a sliding window.
                The default value is 3 years * 364 days, namely 1095 days.

            normalize : str
                Type of scaling to be performed.
                Possible values are
                    - ``'Norm'``
                    - ``'Norm1'``
                    - ``'Std'``
                    - ``'Median'``
                    - ``'Invariant'``
                The default value is ``'Invariant'``.

            criterion : str
                Criterion used to select the best value of the L1 regularization parameter
                by making a trade-off between the goodness of fit and the complexity of the model.
                A good model should explain well the data while being simple.
                Possible values are
                    - ``'aic'``
                    - ``'bic'``.
                The default value is ``'aic'``

            max_iter : int
                Maximum number of iterations for the LASSO algorithm to train.
                It can be used for early stopping.
                The default value is 2500.

            price_lags : tuple
                Daily lags of the response_col variable used for training and forecasting.
                The default values are (1, 2, 3, 7)

            exog_lags : tuple
                Daily lags of the exogenous variables used for training and forecasting.
                The default values are (0, 1, 2)

            dow_dummies : bool
                Whether the day of week dummy variables are used for training and forecasting or not.
                The default is True

            daily_delivery_period_numbers : int
                Number of delivery periods in a day.
                The default value is 24.
        """
        # Set calibration window in days
        self.calibration_window = calibration_window

        # Set data normalization type
        self.normalize = normalize

        # Set the criterion used to select the best model
        self.criterion = criterion

        # Set the maximum number of iterations for the LASSO algorithm
        self.max_iter = max_iter

        # Set the lags of the response_col variable used for training and forecasting
        self.price_lags = price_lags

        # Set the lags of the exogenous variables used for training and forecasting
        self.exog_lags = exog_lags

        # Set whether the day_of_week dummy variables are used for training and forecasting or not
        self.dow_dummies = dow_dummies

        # Set the number of delivery periods in a day
        self.daily_delivery_period_numbers = daily_delivery_period_numbers

        # Set the a priori non defined parameters as None
        self.models = None
        self.scalerX = None
        self.scalerY = None
        self.delivery_day_date = None

    # Ignore convergence warnings from scikit-learn LASSO module
    @ignore_warnings(category=ConvergenceWarning)
    def recalibrate(self, x_train, y_train):
        """ Recalibrate as many LEAR models as many delivery periods are in a delivery day
        using the training dataset.

        Parameters
        ----------
            x_train : pandas.DataFrame
                Explanatory variables in the training dataset within the calibration window.
                It should be of size *[n, m]* where *n* is the number of calibration days
                within in the training dataset and is *m* the number of predictor features.

            y_train : pandas.DataFrame
                Response variables in the training dataset within the calibration window.
                It should be of size *[n, m]* where *n* is the number of calibration days
                within in the training dataset, and *m* is the number of delivery periods within a day.
        """

        # Rescale the response_col variable of the training set
        [y_train_np], self.scalerY = scaling(datasets=[y_train.to_numpy()], normalize=self.normalize)

        # Rescale all explanatory variables of the training set except dummies (the last 7 features/columns)
        x_train_dummies = x_train.filter(like='dayofweek', axis=1).to_numpy()
        x_train_wo_dummies = x_train.filter(regex=r'^((?!dayofweek).)*$', axis=1).to_numpy()
        [x_train_wo_dummies], self.scalerX = scaling(datasets=[x_train_wo_dummies], normalize=self.normalize)
        x_train_np = np.concatenate((x_train_wo_dummies, x_train_dummies), axis=1)

        # iterate over hours of a general day (products) and calibrate LEAR model for each hour
        self.models = {}
        for p in range(self.daily_delivery_period_numbers):

            # Instantiate a LassoLarsIC (Lasso with Least Angle Regression Shrinkage) model object.
            # The AIC or BIC criteria are useful to select the value of the regularization parameter
            # by making a trade-off between the goodness of fit and the complexity of the model.
            # A good model should explain well the data while being simple.
            param_model = LassoLarsIC(criterion=self.criterion, max_iter=self.max_iter, verbose=True)

            # Fit the Lasso model with 'Least Angle Regression Shrinkage' on the training dataset
            # to the get the best value of the 'lambda' hyperparameter of L1 regularization.
            # The 'lambda' hyperparameter controls the degree of sparsity of the estimated coefficients.
            param_model.fit(X=x_train_np, y=y_train_np[:, p])

            # print the lambda hyperparameter (unfortunately, the lambda keyword is upfront reserved by python)
            print('p{0} L1 lambda parameter: {1} ({2} iterations)'.
                  format(p, param_model.alpha_, param_model.n_iter_))

            # Instantiate a Lasso model object using the best value of the 'lambda' for L1 regularization.
            lear = Lasso(max_iter=self.max_iter, alpha=param_model.alpha_)

            # Fit the Lasso Estimated AutoRegressive model on the training dataset.
            lear.fit(X=x_train_np, y=y_train_np[:, p])

            # assign the fitted model to the dictionary
            self.models[p] = lear

    def predict(self, x_test):
        """ Calculates as many predictions as many delivery periods are in a delivery day
        using the related explanatory variables.

        Parameters
        ----------
            x_test : pandas.DataFrame
                Explanatory variables of the test period.

        Returns
        -------
            numpy.ndarray
                An array containing the day-ahead electricity price predictions
                for each product of the in scope day(s).
        """
        # Predefine predicted prices
        y_pred = np.zeros(shape=(x_test.shape[0], self.daily_delivery_period_numbers))

        # Rescale all inputs except 'dayofweek' dummies (7 last features)
        x_test_dummies = x_test.filter(like='dayofweek', axis=1).to_numpy()
        x_test_wo_dummies = x_test.filter(regex=r'^((?!dayofweek).)*$', axis=1).to_numpy()
        x_test_wo_dummies = self.scalerX.transform(dataset=x_test_wo_dummies)
        x_test_np = np.concatenate((x_test_wo_dummies, x_test_dummies), axis=1)

        # Predict the hourly price of the in scope day-ahead using a recalibrated LEAR model
        for p in range(self.daily_delivery_period_numbers):

            # Predict the response_col variable based on the explanatory variables
            y_pred[:, p] = self.models[p].predict(x_test_np)

        # Inverse transforms the predictions
        y_pred = self.scalerY.inverse_transform(dataset=y_pred)

        return y_pred

    def _pivot_lag_extend(self, df):
        """ Internal function to turn the long dataframe wider (from hourly to daily resolution),
        to calculate lagged values of variables as new columns,
        and to add day of week dummies as new columns.

        Parameters
        ----------
            df : pandas.DataFrame
                A 'long' dataframe containing the electricity day-ahead prices and some exogenous variables
                in hourly resolution.

        Returns
        -------
            pandas.DataFrame
                A wide dataframe containing the original and lagged electricity day-ahead prices
                and the original and lagged exogenous variables in daily resolution.
                Each hours' value is put in different columns.
        """
        # Check that the first index of the DataFrame corresponds with the hour 00:00
        if df.index[0].hour != 0:
            raise Exception('The first index in the dataframe does not correspond with the hour 00:00.')

        # Detect the names of Exogenous inputs
        exogenous_inputs = [col for col in df.columns if 'exogenous' in col.casefold()]

        # Create a list which first element is a dataframe of day-ahead electricity prices
        shifted_responses = [df[['Price']]]

        # Iterate over the price_lags to calculate accordingly shifted response_col variables for each datetime index.
        # Each lagged value composes a new one-column temporary dataframe.
        for past_day in self.price_lags:
            shifted_df = pd.DataFrame(data=df.Price.values,
                                      index=df.index + pd.Timedelta(days=past_day),
                                      columns=['shifted_{0}d'.format(past_day)])
            shifted_responses.append(shifted_df)

        # Iterate over the exog_lags to calculate accordingly shifted exogenous inputs for each datetime index.
        # Each lagged value composes a new one-column dataframe.
        # Attention!
        # There is a zero lag for exogenous inputs, which means that no lagged values
        # remain among the exogenous inputs as well.
        for past_day in self.exog_lags:
            # iterate over each exogenous input
            for exog in exogenous_inputs:
                shifted_df = pd.DataFrame(data=df[exog].values,
                                          index=df.index + pd.Timedelta(days=past_day),
                                          columns=['{0}_shifted_{1}d'.format(exog, past_day)])
                shifted_responses.append(shifted_df)

        # Add column-wise up the new lagged values to form a new lagged dataframe
        # of response and explanatory variables for each datetime index.
        df_lagged = pd.concat(objs=shifted_responses, axis=1, join='inner')
        del shifted_responses

        # Pivot the training DataFrames wider by the hour part of the index
        if self.daily_delivery_period_numbers == 24:
            df_lagged['period_of_the_day'] = ['p' + str(ind.hour).zfill(2) for ind in df_lagged.index]
        else:
            df_lagged['period_of_the_day'] = ['p' + str(ind.hour).zfill(2) + str(ind.minute).zfill(2)
                                              for ind in df_lagged.index]
        df_lagged_wide = pd.pivot_table(data=df_lagged,
                                        index=df_lagged.index.date,
                                        columns='period_of_the_day', aggfunc='mean', sort=False)
        df_lagged_wide.index = pd.to_datetime(df_lagged_wide.index)  # convert back to datetime
        df_lagged_wide.index.name = 'date'
        df_lagged_wide.columns.name = 'variable'
        df_lagged_wide.columns = ['{1}_{0}'.format(i, j) for i, j in df_lagged_wide.columns.to_list()]
        df_lagged_wide.columns = [col.rstrip('_Price') for col in df_lagged_wide.columns]

        # If demanded, add the day-of-week dummy variables that depend on the delivery day of the week,
        # where Monday is 0 and Sunday is 6.
        if self.dow_dummies:
            for dow in range(7):
                df_lagged_wide['dayofweek_{0}'.format(dow)] = 0
                df_lagged_wide.loc[df_lagged_wide.index.dayofweek == dow, 'dayofweek_{0}'.format(dow)] = 1

        return df_lagged_wide

    def recalibrate_and_forecast_next_day(self, df):
        """ Easy-to-use interface for daily recalibration and forecasting of the LEAR model.

        The method receives a pandas dataframe and a date.
        First of all, it recalibrates the model using data up to the day before ``delivery_day_date``
        and makes a prediction for day ``delivery_day_date``.

        Parameters
        ----------
            df : pandas.DataFrame
                The long dataframe of historical data containing day_ahead electricity prices
                and the values of *N* exogenous inputs.
                The last (in-focus) day's day-ahead electricity price data is set to NaN a priori.
                The index of the dataframe should be timestamps with evenly distributed frequency
                of electricity delivery periods. The column names should follow this convention:
                ``['Price', 'Exogenous_1', 'Exogenous_2', ..., 'Exogenous_N']``.

        Returns
        -------
            numpy.ndarray
                The prediction of day-ahead prices.
        """
        # extract the starting datetime of the in-focus delivery day
        self.delivery_day_date = df.loc[df.Price.isnull()].index[0]

        # calculate the starting datetime of the calibration window
        calibration_start_at = (self.delivery_day_date - pd.Timedelta(days=self.calibration_window))

        # turn the dataframe into a wide format (from hourly to daily)
        # and add the lags of variables as new columns
        # and add day of week dummies if requested
        df_wide = self._pivot_lag_extend(df=df)

        # limit the training dataset starting point according to the calibration window
        # and end point just before the in-focus delivery day
        df_train_resliced = df_wide.loc[(df_wide.index >= calibration_start_at) &
                                        (df_wide.index < self.delivery_day_date)]

        # limit the test dataset on the in-focus delivery day
        df_test_resliced = df_wide.loc[df_wide.index == self.delivery_day_date]

        # Column-wise slice the X, Y train dataframes for training
        # x_train: the wide dataframe of explanatory variables in the training period
        # y_train: the wide dataframe of daily target variables in the training period
        x_train = df_train_resliced.filter(regex=r'^(?!p[0-9]{,2}$).*', axis=1)
        y_train = df_train_resliced.filter(regex=r'p[0-9]{,2}$', axis=1)

        # Column-wise slice the X test dataframe for prediction
        # x_test: the wide dataframe of explanatory variables in the test period
        x_test = df_test_resliced.filter(regex=r'^(?!p[0-9]{,2}$).*', axis=1)

        # Recalibrate the in-focus delivery day related LEAR model using the values of
        # all the explanatory and response variables within the calibration window.
        self.recalibrate(x_train=x_train, y_train=y_train)

        # Predict target variable values on the test dataset
        y_pred = self.predict(x_test=x_test)

        return y_pred


def evaluate_lear_in_test_dataset(path_datasets_folder=os.path.join('..', '..', 'examples', 'datasets'),
                                  path_recalibration_folder=os.path.join('..', '..', 'examples', 'experimental_files'),
                                  dataset=None,
                                  index_col=None,
                                  response_col=None,
                                  sep=',',
                                  decimal='.',
                                  date_format='ISO8601',
                                  encoding='utf-8',
                                  calibration_window=364 * 3,
                                  years_test=2,
                                  begin_test_date=None,
                                  end_test_date=None,
                                  price_lags=(1, 2, 3, 7),
                                  exog_lags=(0, 1, 2),
                                  dow_dummies=True,
                                  save_frequency=10,
                                  index_tz=None,
                                  market_tz=None,
                                  intended_freq='1h',
                                  summary=False):
    """ Easy evaluation of the `LEAR` model in a test dataset using daily recalibration.

    The test dataset is defined by a market name and the test dates.
    The function generates the test and training datasets,
    and evaluates a LEAR model considering daily recalibration.

    Parameters
    ----------
        path_datasets_folder : str
            The path of the local folder where the input datasets are stored.
            In case the folder does not exist yet, this will be the path of the folder
            where the input datasets will be downloaded locally.

        path_recalibration_folder : str
            The path of that local folder where the experiment's result dataset file will be saved.

        dataset : str
            The filename (w/o "csv" extension) of the input dataset/market under study.
            If one of the standard open-access benchmark datasets referred,
            such as "PJM", "NP", "BE", "FR", or "DE", then that will automatically be
            downloaded from zenodo.org into the ``path_datasets_folder``.
            In any other case, the input csv dataset should be placed in advance into the ``path_datasets_folder``.

        index_col : int | str | None
            A column of the input dataset to use as datetime index,
            denoted either by column label or column index.
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        response_col : int |str | None
            A column of the input dataset to use as response variable,
            denoted either by column label or column index.
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        sep : str
            Delimiter of the input dataset.
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        decimal : str
            Decimal point of the input dataset.
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        date_format : str
            Date format to use parsing dates of the input dataset.
            For details see here: https://en.wikipedia.org/wiki/ISO_8601
            and here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        encoding : str
            Encoding of the input csv file.
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        calibration_window : int
            The number of days (a year is considered as 364 days)
            that are used in the training dataset for recalibration.
            Limits the training dataset starting point in a sliding window.

        years_test : int | None
            The number of years (a year is 364 days) in the test dataset.
            It is only used if the arguments ``begin_test_date`` and ``end_test_date`` are not provided.

        begin_test_date : datetime.datetime | str | None
            Optional parameter, which defines the starting timestamp of the test dataset.
            Used in combination with the argument ``end_test_date``.
            If either of them is not provided, then the test dataset will be filtered upon
            the ``years_test`` argument.
            The ``begin_test_date`` should either be string in "%Y-%m-%d 00:00:00" format,
            or datetime object.

        end_test_date : datetime.datetime | str | None
            Optional parameter, which defines the closing timestamp of the test dataset.
            Used in combination with the argument ``begin_test_date``.
            If either of them is not provided, then the test dataset will be filtered upon
            the ``years_test`` argument.
            The ``end_test_date`` should either be string in ``"%Y-%m-%d 23:00:00"`` format,
            or a datetime object.

        price_lags : tuple
                Daily lags of the response_col variable used for training and forecasting.
                The default values are (1, 2, 3, 7)

        exog_lags : tuple
            Daily lags of the exogenous variables used for training and forecasting.
            The default values are (0, 1, 2)

        dow_dummies : bool
            Whether the day of week dummy variables are used for training and forecasting or not.
            The default is True

        save_frequency : int
            The daily frequency of saving the results into the ``path_recalibration_folder``.
            The default value is 5.

        index_tz : str | datetime.tzinfo
            In case of non-canonical datasets, the timezone in which the index datetime values are expressed.
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        market_tz : str | datetime.tzinfo
            In case of non-canonical datasets, the timezone in which the electricity market takes place.
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        intended_freq : str
            In case of non-canonical datasets, the intended frequency of the datetime index.
            It must take one of the following four values ``'1h'`` for 1 hour, ``'30min'`` for 30 minutes,
            ``'15min'`` for 15 minutes, or ``'5min'`` for 5 minutes, (these are the four standard values in
            day-ahead electricity markets).
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        summary : bool
            Whether to print a summary of the resulting DataFrames.

    Returns
    -------
        pandas.DataFrame
            A dataframe with all the predictions related to the test dataset.
            As a side effect, the result will be saved into the ``path_recalibration_folder`` as well.
    """

    # Check if the provided directory for recalibration exists and if not create it
    os.makedirs(name=path_recalibration_folder, exist_ok=True)

    # import the whole dataset
    df_all = read_data(path=path_datasets_folder, dataset=dataset, index_col=index_col, response_col=response_col,
                       sep=sep, decimal=decimal, date_format=date_format, encoding=encoding, index_tz=index_tz,
                       market_tz=market_tz, intended_freq=intended_freq, summary=summary)

    # split into train and test data
    df_train, real_values = split_data(df=df_all, years_test=years_test, begin_test_date=begin_test_date,
                                       end_test_date=end_test_date, summary=summary)

    # calculate daily delivery period numbers
    daily_delivery_period_numbers = {'h': 24, '30min': 48, '15min': 96, '5min': 288}.\
        get(real_values.index.inferred_freq, 24)

    # define unique file name to save the forecast
    forecast_file_name = ('LEAR_forecast_dat{0}_YT{1}_CW{2}_{3}.csv'.
                          format(str(dataset), str(years_test), str(calibration_window),
                                 datetime.now().strftime('%Y%m%d_%H%M%S')))

    # compose the whole path of the forecast file name
    forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)

    # Create a DataFrame from the real day-ahead price values of the test period,
    # where the dates are the rows and the hours are the columns, and the values are the prices.
    if daily_delivery_period_numbers == 24:
        real_values.loc[:, ['period_of_the_day']] = ['p' + str(ind.hour).zfill(2) for ind in real_values.index]
    else:
        real_values.loc[:, ['period_of_the_day']] = ['p' + str(ind.hour).zfill(2) + str(ind.minute).zfill(2)
                                                     for ind in real_values.index]
    real_values = pd.pivot_table(data=real_values, values='Price',
                                 index=real_values.index.date,
                                 columns='period_of_the_day', aggfunc='mean', sort=False)

    # adjust index names
    real_values.index.name = 'date'

    # convert index column to datetime format and set frequency
    real_values.index = pd.to_datetime(arg=real_values.index)
    real_values.index.freq = real_values.index.inferred_freq

    # define an empty (wide) dataframe for prospective forecasts
    forecast = pd.DataFrame(index=real_values.index, columns=real_values.columns)

    # instantiate a LEAR model object with the given calibration window
    model = LEAR(calibration_window=calibration_window, normalize='Invariant', criterion='aic', max_iter=2500,
                 price_lags=price_lags, exog_lags=exog_lags, dow_dummies=dow_dummies,
                 daily_delivery_period_numbers=daily_delivery_period_numbers)

    # for loop over the dates of the test period
    for key, delivery_date_start in enumerate(forecast.index):

        # calculate the last timestamp of the in-focus delivery day
        # this is the last one before the next delivery day's start
        delivery_date_end = df_all.index[df_all.index < delivery_date_start + pd.Timedelta(days=1)][-1]

        # slice a new (long) dataframe from the whole data up to the in-focus delivery day's end
        data_available = df_all.loc[:delivery_date_end].copy()

        # set the real day-ahead electricity prices of the in-focus delivery day to NaN
        # in the data_available (long) dataframe
        data_available.loc[delivery_date_start:, 'Price'] = np.NaN

        # Recalibrate the model with the most up-to-date available data
        # and making a prediction for the current_date
        y_pred = model.recalibrate_and_forecast_next_day(df=data_available)

        # fill up the forecast (wide) dataframe with the current_date's predictions
        forecast.loc[delivery_date_start, :] = y_pred

        # compute metrics up to current_date
        mae = np.mean(a=MAE(p_real=forecast.loc[:delivery_date_start].values.squeeze(),
                            p_pred=real_values.loc[:delivery_date_start].values.squeeze()))
        smape = np.mean(a=sMAPE(p_real=forecast.loc[:delivery_date_start].values.squeeze(),
                                p_pred=real_values.loc[:delivery_date_start].values.squeeze())) * 100

        # print error information
        print('{0} - sMAPE: {1:.2f}%  |  MAE: {2:.3f}'.format(str(delivery_date_start)[:10], smape, mae))

        # save the forecasts in save_frequency days chunks
        if (key + 1) % save_frequency == 0 or (key + 1) == len(forecast.index):
            forecast.to_csv(path_or_buf=forecast_file_path, sep=sep, decimal=decimal, date_format='%Y-%m-%d',
                            encoding=encoding, index=True, index_label='date')

    return forecast


if __name__ == '__main__':
    # These codes below will only be executed if this script is run as the main program.
    predictions = evaluate_lear_in_test_dataset(dataset='DE', calibration_window=364 * 1,
                                                begin_test_date='2017-12-01 00:00:00',
                                                end_test_date='2017-12-10 23:00')
    print(predictions)
