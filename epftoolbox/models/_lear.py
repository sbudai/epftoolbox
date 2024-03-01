"""
Classes and functions to implement the LEAR model for electricity price forecasting
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LassoLarsIC, Lasso
from epftoolbox.data import read_and_split_data, scaling
from epftoolbox.evaluation import MAE, sMAPE
# noinspection PyProtectedMember
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime


class LEAR(object):
    """ Class to build a LEAR model, recalibrate it, and use it to predict DA electricity prices. """

    def __init__(self, calibration_window=364 * 3, normalize='Invariant', criterion='aic', max_iter=2500,
                 price_lags=(1, 2, 3, 7), exog_lags=(0, 1, 2), dow_dummies=True):
        """ Instantiate a LEAR model object.

        Parameters
        ----------
            calibration_window : int
                Calibration window (in days) for the model training.
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

        # Set the a priori non defined parameters as None
        self.models = None
        self.scalerX = None
        self.scalerY = None

    # Ignore convergence warnings from scikit-learn LASSO module
    @ignore_warnings(category=ConvergenceWarning)
    def recalibrate(self, x_train, y_train):
        """ Function to recalibrate the LEAR model.
        It uses a training DataFrames (x_train, y_train) for model recalibration.

        Parameters
        ----------
            x_train : pandas.DataFrame
                Predictors in training dataset.
                It should be of size *[n,m]* where *n* is the number of days
                in the training dataset and *m* the number of input features

            y_train : pandas.DataFrame
                Response variables in training dataset.
                It should be of size *[n,24]* where *n* is the number of days
                in the training dataset and 24 are the 24 prices of each day.

        Returns
        -------
            numpy.ndarray
                The prediction of day-ahead prices after recalibrating the model.
        """

        # Rescale the response_col variable of the training set
        [y_train_np], self.scalerY = scaling(datasets=[y_train.to_numpy()], normalize=self.normalize)

        # Rescale all explanatory variables of the training set except dummies (the last 7 features/columns)
        tf_cols = [col for col in x_train.columns if 'dayofweek' not in col.casefold()]
        no_tf_cols = [col for col in x_train.columns if 'dayofweek' in col.casefold()]
        [x_train_wo_dummies], self.scalerX = scaling(datasets=[x_train[tf_cols].to_numpy()], normalize=self.normalize)
        x_train_np = np.concatenate((x_train_wo_dummies, x_train[no_tf_cols].to_numpy()), axis=1)
        del x_train_wo_dummies

        # iterate over hours of a general day (products) and calibrate LEAR model for each hour
        self.models = {}
        for h in range(24):

            # Instantiate a LassoLarsIC (Lasso with Least Angle Regression Shrinkage) model object.
            # The AIC or BIC criteria are useful to select the value of the regularization parameter
            # by making a trade-off between the goodness of fit and the complexity of the model.
            # A good model should explain well the data while being simple.
            param_model = LassoLarsIC(criterion=self.criterion, max_iter=self.max_iter, verbose=True)

            # Fit the Lasso model with 'Least Angle Regression Shrinkage' on the training dataset
            # to the get the best value of the 'lambda' hyperparameter of L1 regularization.
            # The 'lambda' hyperparameter controls the degree of sparsity of the estimated coefficients.
            param_model.fit(X=x_train_np, y=y_train_np[:, h])

            # print the lambda hyperparameter (unfortunately, the lambda keyword is upfront reserved by python)
            print('h{0} L1 lambda parameter: {1} ({2} iterations)'.
                  format(h, param_model.alpha_, param_model.n_iter_))

            # Instantiate a Lasso model object using the best value of the 'lambda' for L1 regularization.
            lear = Lasso(max_iter=self.max_iter, alpha=param_model.alpha_)

            # Fit the Lasso Estimated AutoRegressive model on the training dataset.
            lear.fit(X=x_train_np, y=y_train_np[:, h])

            # assign the fitted model to the dictionary
            self.models[h] = lear

    def predict(self, x_test):
        """ Function that makes a prediction using some given inputs.

        Parameters
        ----------
            x_test : pandas.DataFrame
                Explanatory variables of the predictions.

        Returns
        -------
            numpy.ndarray
                An array containing the price predictions
                for each product of the in scope day(s).
        """

        # Predefining predicted prices
        y_pred = np.zeros(shape=(x_test.shape[0], 24))

        # Rescaling all inputs except 'dayofweek' dummies (7 last features)
        tf_cols = [col for col in x_test.columns if 'dayofweek' not in col.casefold()]
        no_tf_cols = [col for col in x_test.columns if 'dayofweek' in col.casefold()]
        x_test_np = np.concatenate((self.scalerX.transform(dataset=x_test[tf_cols].to_numpy()),
                                    x_test[no_tf_cols].to_numpy()), axis=1)

        # Predicting the in scope hour day-ahead prices using a recalibrated LEAR model
        for h in range(24):

            # Predict the response_col variable based on the explanatory variables
            y_pred[:, h] = self.models[h].predict(X=x_test_np)

        # Inverse transforms the predictions
        y_pred = self.scalerY.inverse_transform(dataset=y_pred)

        return y_pred

    def _build_and_split_x_y(self, df_train, df_test, date_test=None):
        """ Internal function that generates the X, Y arrays for training and testing based on pandas dataframes

        Parameters
        ----------
            df_train : pandas.DataFrame
                Pandas dataframe containing the training data

            df_test : pandas.DataFrame
                Pandas dataframe containing the test data

            date_test : datetime.datetime
                If given, then the test dataset is only built for that date

        Returns
        -------
            list[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
                A list of 3 pandas DataFrames containing the explanatory and response_col variable values
                in from the training dataset, the response_col variable values for the test dataset.
        """
        # Check that the first index of the DataFrames corresponds with the hour 00:00
        if df_train.index[0].hour != 0:
            raise Exception('The first index in df_train does not correspond with the hour 00:00.')
        if df_test.index[0].hour != 0:
            raise Exception('The first index in df_test does not correspond with the hour 00:00.')

        # Detect the names of Exogenous inputs
        exogenous_inputs = [col for col in df_train.columns if 'exogenous' in col.casefold()]

        # extract the last index of the training set and the last index of the test set
        last_train_index = df_train.index[-1]
        last_test_index = df_test.index[-1]

        shifted_train_responses = []
        shifted_test_responses = []

        # iterate over the price_lags to calculate accordingly shifted response_col variables
        for past_day in self.price_lags:
            shifted_df_train = pd.DataFrame(data=df_train.to_dict(orient='list')['Price'],
                                            index=df_train.index + pd.Timedelta(days=past_day),
                                            columns=['response_shifted_{0}d'.format(past_day)])
            shifted_df_train = shifted_df_train.loc[shifted_df_train.index <= last_train_index, :]
            shifted_train_responses.append(shifted_df_train)

            shifted_df_test = pd.DataFrame(data=df_test.to_dict(orient='list')['Price'],
                                           index=df_test.index + pd.Timedelta(days=past_day),
                                           columns=['response_shifted_{0}d'.format(past_day)])
            shifted_df_test = shifted_df_test.loc[shifted_df_test.index >= date_test, :]
            shifted_df_test = shifted_df_test.loc[shifted_df_test.index <= last_test_index, :]
            shifted_test_responses.append(shifted_df_test)

        # iterate over the exog_lags to calculate accordingly shifted exogenous inputs
        for past_day in self.exog_lags:
            # iterate over each exogenous input
            for exog in exogenous_inputs:
                shifted_df_train = pd.DataFrame(data=df_train.to_dict(orient='list')[exog],
                                                index=df_train.index + pd.Timedelta(days=past_day),
                                                columns=['{0}_shifted_{1}d'.format(exog, past_day)])
                shifted_df_train = shifted_df_train.loc[shifted_df_train.index <= last_train_index, :]
                shifted_train_responses.append(shifted_df_train)

                shifted_df_test = pd.DataFrame(data=df_test.to_dict(orient='list')[exog],
                                               index=df_test.index + pd.Timedelta(days=past_day),
                                               columns=['{0}_shifted_{1}d'.format(exog, past_day)])
                shifted_df_test = shifted_df_test.loc[shifted_df_test.index >= date_test, :]
                shifted_df_test = shifted_df_test.loc[shifted_df_test.index <= last_test_index, :]
                shifted_test_responses.append(shifted_df_test)

        # Bind column-wise the shifted train and test DataFrames separately
        df_x_train = pd.concat(objs=shifted_train_responses, axis=1, join='inner')
        df_x_test = pd.concat(objs=shifted_test_responses, axis=1, join='inner')
        del shifted_train_responses, shifted_test_responses

        # Pivot the train DataFrames wider by the hour part of the index
        df_x_train['column_hour'] = ['h' + h for h in df_x_train.index.strftime('%H').astype(int).astype(str)]
        df_x_train = pd.pivot_table(data=df_x_train,
                                    index=df_x_train.index.date,
                                    columns='column_hour', aggfunc='mean', sort=False)
        df_x_train.index = pd.to_datetime(df_x_train.index)
        df_x_train.columns = ['{1}_{0}'.format(i, j) for i, j in df_x_train.columns.to_list()]

        # Pivot the test DataFrames wider by the hour part of the index
        df_x_test['column_hour'] = ['h' + h for h in df_x_test.index.strftime('%H').astype(int).astype(str)]
        df_x_test = pd.pivot_table(data=df_x_test,
                                   index=df_x_test.index.date,
                                   columns='column_hour', aggfunc='mean', sort=False)
        df_x_test.index = pd.to_datetime(df_x_test.index)
        df_x_test.columns = ['{1}_{0}'.format(i, j) for i, j in df_x_test.columns.to_list()]

        # If demanded, add the dummy variables that depend on the day of the week,
        # where Monday is 0 and Sunday is 6.
        if self.dow_dummies:
            for dow in range(7):
                df_x_train.loc[df_x_train.index.dayofweek == dow, 'dayofweek_{0}'.format(dow)] = 1
                df_x_train.loc[df_x_train.index.dayofweek != dow, 'dayofweek_{0}'.format(dow)] = 0
                df_x_test.loc[df_x_test.index.dayofweek == dow, 'dayofweek_{0}'.format(dow)] = 1
                df_x_test.loc[df_x_test.index.dayofweek != dow, 'dayofweek_{0}'.format(dow)] = 0

        # Extract the response_col variable values from the train Dataframe
        # and pivot it wider by the hour part of the index
        df_y_train = df_train.loc[df_x_train.index[0]:, ['Price']]
        df_y_train['column_hour'] = ['h' + h for h in df_y_train.index.strftime('%H').astype(int).astype(str)]
        df_y_train = pd.pivot_table(data=df_y_train,
                                    index=df_y_train.index.date,
                                    columns='column_hour', aggfunc='mean', sort=False)
        df_y_train.index = pd.to_datetime(df_y_train.index)
        df_y_train.columns = ['{1}_{0}'.format(i, j) for i, j in df_y_train.columns.to_list()]

        return df_x_train, df_y_train, df_x_test

    def recalibrate_and_forecast_next_day(self, df, next_day_date):
        """ Easy-to-use interface for daily recalibration and forecasting of the LEAR model.

        The function receives a pandas dataframe and a date. Usually, the data should
        correspond with the date of the next day when using for daily recalibration.

        Parameters
        ----------
            df : pandas.DataFrame
                Dataframe of historical data containing prices and *N* exogenous inputs.
                The index of the dataframe should be dates with hourly frequency. The columns
                should have the following names ``['Price', 'Exogenous 1', 'Exogenous 2', ..., 'Exogenous N']``.

            next_day_date : datetime.datetime
                Date of the day-ahead auction.

        Returns
        -------
            numpy.ndarray
                The prediction of day-ahead prices.
        """

        # Define the new training dataset which lasts till the day of interest
        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]

        # Limit the training dataset starting point according to the calibration window
        df_train = df_train.iloc[-self.calibration_window * 24:]

        # Define the test dataset as the last 2 weeks plus the day of interest,
        # to be able to build the necessary input features.
        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        # Generating X, Y pairs for predicting prices
        x_train, y_train, x_test = self._build_and_split_x_y(df_train=df_train, df_test=df_test,
                                                             date_test=next_day_date)

        # Recalibrating the LEAR model for each hour on the training dataset
        self.recalibrate(x_train=x_train, y_train=y_train)

        # Predict target variable values on the test dataset
        y_pred = self.predict(x_test=x_test)

        return y_pred


def evaluate_lear_in_test_dataset(path_datasets_folder=os.path.join('..', '..', 'examples', 'datasets'),
                                  path_recalibration_folder=os.path.join('..', '..', 'examples', 'experimental_files'),
                                  dataset=None,
                                  response=None,
                                  calibration_window=364 * 3,
                                  years_test=2,
                                  begin_test_date=None,
                                  end_test_date=None,
                                  price_lags=(1, 2, 3, 7),
                                  exog_lags=(0, 1, 2),
                                  dow_dummies=True,
                                  save_frequency=5):
    """ Function for easy evaluation of the `LEAR` model in a test dataset using daily recalibration.

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

        response : str
            The name of the target variable in the dataset.

        calibration_window : int
            The number of days (a year is 364 days) used in the training dataset for recalibration.

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

    Returns
    -------
        pandas.DataFrame
            A dataframe with all the predictions related to the test dataset.
            As a side effect, the result will be saved into the ``path_recalibration_folder`` as well.
    """

    # Checking if provided directory for recalibration exists and if not create it
    os.makedirs(name=path_recalibration_folder, exist_ok=True)

    # Defining train and testing data
    df_train, df_test = read_and_split_data(path=path_datasets_folder,
                                            dataset=dataset,
                                            response_col=response,
                                            years_test=years_test,
                                            begin_test_date=begin_test_date,
                                            end_test_date=end_test_date)

    # Defining unique file name to save the forecast
    forecast_file_name = ('LEAR_forecast_dat{0}_YT{1}_CW{2}_{3}.csv'.
                          format(str(dataset), str(years_test), str(calibration_window),
                                 datetime.now().strftime('%Y%m%d_%H%M%S')))

    # Compose the whole path of the forecast file name
    forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)

    # Create a DataFrame from the real day-ahead price values of the test period
    # where the dates are the rows and the hours are the columns, and the values are the prices
    real_values = df_test.loc[:, ['Price']]
    real_values['column_hour'] = ['h' + h for h in real_values.index.strftime('%H').astype(int).astype(str)]
    real_values = pd.pivot_table(data=real_values, values='Price',
                                 index=real_values.index.date,
                                 columns='column_hour', aggfunc='mean', sort=False)

    # convert index column to datetime format
    real_values.index = pd.to_datetime(real_values.index)

    # Define an empty forecast DataFrame
    forecast = pd.DataFrame(index=real_values.index, columns=real_values.columns)

    # instantiate a LEAR model object with the given calibration window
    model = LEAR(calibration_window=calibration_window,
                 normalize='Invariant', criterion='aic', max_iter=2500,
                 price_lags=price_lags, exog_lags=exog_lags, dow_dummies=dow_dummies)

    # For loop over the recalibration dates
    for key, current_date in enumerate(forecast.index):

        # calculate the last timestamp of the current date
        current_date_end = current_date + pd.Timedelta(hours=23)

        # For simulation purposes, we assume that the available data is the train data
        # plus the test data up to current date's end
        data_available = pd.concat(objs=[df_train, df_test.loc[:current_date_end, :]],
                                   axis=0)

        # We set the real prices for current date to NaN in available_data DataFrame
        data_available.loc[current_date:current_date_end, 'Price'] = np.NaN

        # Recalibrate the model with the most up-to-date available data
        # and making a prediction for the current_date
        y_pred = model.recalibrate_and_forecast_next_day(df=data_available, next_day_date=current_date)
        # Save the current_date's predictions
        forecast.loc[current_date, :] = y_pred

        # Computing metrics up to current_date
        mae = np.mean(a=MAE(p_real=forecast.loc[:current_date].values.squeeze(),
                            p_pred=real_values.loc[:current_date].values.squeeze()))
        smape = np.mean(a=sMAPE(p_real=forecast.loc[:current_date].values.squeeze(),
                                p_pred=real_values.loc[:current_date].values.squeeze())) * 100

        # Print information
        print('{0} - sMAPE: {1:.2f}%  |  MAE: {2:.3f}'.format(str(current_date)[:10], smape, mae))

        # Save the forecasts in save_frequency days chunks
        if (key + 1) % save_frequency == 0 or (key + 1) == len(forecast.index):
            forecast.to_csv(path_or_buf=forecast_file_path, sep=';', index=True, index_label='date', encoding='utf-8')

    return forecast


if __name__ == '__main__':
    # These codes below will only be executed if this script is run as the main program.
    predictions = evaluate_lear_in_test_dataset(
        dataset='DE',
        response='Price',
        years_test=None,
        calibration_window=364 * 1,
        begin_test_date='01/12/2017 00:00',
        end_test_date='10/12/2017 23:00')

    print(predictions)
