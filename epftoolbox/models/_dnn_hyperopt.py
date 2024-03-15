"""
Classes and functions to perform hyperparameter and feature selection for the DNN model
for the day-ahead electricity price forecasting.
"""

# Author: Jesus Lago & Sandor Budai

# License: AGPL-3.0 License

import pandas as pd
import numpy as np
import pickle as pc
import os

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from datetime import datetime
from epftoolbox.models import DNNModel, pivot_lag_extend
from epftoolbox.data import scaling
from epftoolbox.data import read_data, split_data
from epftoolbox.evaluation import MAE, sMAPE
from functools import partial


def _build_space(n_layers, n_exogenous_inputs, data_augmentation=None, price_lags=(1, 2, 3, 7),
                 exog_lags=(0, 1, 7)):
    """ Function that generates the hyperparameter/feature search space.

    Parameters
    ----------
        n_layers : int
            Number of layers of the DNN model

        data_augmentation : bool
            Boolean that selects whether a data augmentation technique for DNNs is used.
            Based on empirical results, for some markets data augmentation might
            improve forecasting accuracy at the expense of higher computational costs.
            Data augmentation technique enriches the training dataset with additional
            observations that are derived from the original ones.
            If None, then True and False are selected randomly.

        n_exogenous_inputs : int
            Number of exogenous inputs in the market under study

        price_lags : tuple
                Daily lags of the response_col variable used for training and forecasting.
                The default values are (1, 2, 3, 7)

        exog_lags : tuple
            Daily lags of the exogenous variables used for training and forecasting.
            The default values are (0, 1, 7)

    Returns
    -------
        dict
            Dictionary defining the search space for DNN.
    """
    # define the base of the neural net hyperparameter space
    space = {
        'batch_normalization': hp.choice(label='batch_normalization', options=[False, True]),
        'dropout': hp.uniform('dropout', 0, 1),
        'lr': hp.loguniform('lr', np.log(5e-4), np.log(0.1)),
        'seed': hp.quniform('seed', 1, 1000, 1),
        'neurons1': hp.quniform('neurons1', 50, 500, 1),
        'activation': hp.choice(label='activation',
                                options=["relu", "softplus", "tanh", 'selu', 'LeakyReLU', 'PReLU', 'sigmoid']),
        'init': hp.choice(label='init',
                          options=['Orthogonal', 'lecun_uniform', 'glorot_uniform', 'glorot_normal', 'he_uniform',
                                   'he_normal']),
        'reg': hp.choice(label='reg',
                         options=[
                             {'val': None, 'lambda': 0},
                             {'val': 'l1', 'lambda': hp.loguniform('lambdal1', np.log(1e-5), np.log(1))}
                         ]),
        'scaleX': hp.choice(label='scaleX', options=['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant']),
        'scaleY': hp.choice(label='scaleY', options=['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant'])
    }

    # add further layer neuron parameters if needed
    if n_layers >= 2:
        space['neurons2'] = hp.quniform('neurons2', 25, 400, 1)
    if n_layers >= 3:
        space['neurons3'] = hp.quniform('neurons3', 25, 300, 1)
    if n_layers >= 4:
        space['neurons4'] = hp.quniform('neurons4', 25, 200, 1)
    if n_layers >= 5:
        space['neurons5'] = hp.quniform('neurons5', 25, 200, 1)

    # *** explanatory variables as hyperparameters ***

    # day of week as hyperparameter
    space['In: Day'] = hp.choice(label='In: Day', options=[False, True])

    # data augmentation as hyperparameter
    if data_augmentation is None:
        space['data_augmentation'] = hp.choice(label='data_augmentation', options=[False, True])
    else:
        space['data_augmentation'] = hp.choice(label='data_augmentation', options=[data_augmentation])

    # lagged day-ahead electricity prices as hyperparameters
    for i in price_lags:
        space['In: Price D-{0}'.format(i)] = hp.choice(label='In: Price D-{0}'.format(i), options=[False, True])

    # exogenous explanatory variables and their lagged value as hyperparameters
    for n_ex in range(1, n_exogenous_inputs + 1):
        for exog_lag in exog_lags:
            if exog_lag == 0:
                dict_key = 'In: Exog-{0} D'.format(n_ex)
            else:
                dict_key = 'In: Exog-{0} D-{1}'.format(n_ex, exog_lag)
            space[dict_key] = hp.choice(label=dict_key, options=[False, True])

    return space


def _hyperopt_objective(hyperparameters, trials, trials_file_path, max_evals, n_layers, df_train, df_test,
                        calibration_window, percentage_val, shuffle_train):
    """ Function that defines the hyperparameter optimization objective.

    This function receives a set of hyperparameters as input, trains a DNN using them,
    and returns the performance of the DNN for the selected hyperparameters in a validation
    dataset

    Parameters
    ----------
        hyperparameters : dict
            A dictionary provided by hyperopt indicating whether each hyperparameter/feature is selected or not.

        trials : hyperopt.Trials
            The 'trials' object that stores the results of the hyperparameter optimization runs.

        trials_file_path : str
            The file path to store the 'trials' object

        max_evals : int
            Maximum number of iterations for hyperparameter optimization

        n_layers : int
            Number of layers in the DNN model

        df_train : pandas.DataFrame
            Dataframe containing the training data.

        df_test : pandas.DataFrame
            Dataframe containing the testing data.

        calibration_window : int
            The number of days (a year is considered as 364 days)
            that are used in the training dataset for recalibration.
            Limits the training dataset starting point in a sliding window.

        percentage_val : float
            Percentage of data to be used for validation from the training dataset.

        shuffle_train : bool
            Boolean that selects whether the training dataset rows should be shuffled before ripping off
            the validation dataset.

    Returns
    -------
        dict
            A dictionary summarizing the result of the hyperparameter run
    """
    # extract the starting datetime of the in-focus delivery day(s)
    delivery_day_date = df_test.index[0]

    # Re-define the training dataset based on the calibration window.
    # The calibration window can be given as an external parameter.
    # If value 0 is given, the calibration window is included as a hyperparameter to optimize
    # calculate the starting datetime of the calibration window
    if calibration_window:
        calibration_start_at = (delivery_day_date - pd.Timedelta(days=calibration_window))
    else:
        calibration_start_at = df_train.index[0]

    # append the training and the test dataset into one table
    df = pd.concat([df_train, df_test])

    # turn the appended dataframe into a wide format (from hourly to daily)
    # and add the lags of variables as new columns
    # and add day of week dummies if requested
    x_train, y_train, x_val, y_val, x_test, y_test = pivot_lag_extend(
        df_all=df, features=hyperparameters, data_augmentation=hyperparameters.get('data_augmentation'),
        calibration_start_at=calibration_start_at, begin_test_date=delivery_day_date, end_test_date=df_test.index[-1],
        percentage_val=percentage_val, shuffle_train=shuffle_train, hyperoptimization=True
    )

    # Write the pickled representation of the 'trials' object (which stores the hyperparameter optimization runs)
    # into a file.
    pc.dump(obj=trials, file=open(file=trials_file_path, mode="wb"))

    # if the 'trials' object has already had values, then print them
    if len([i for i in trials.losses() if i is not None]):
        print('\n\t~~~ after {0}/{1} iterations'.format(len(trials.losses()) - 1, max_evals))
        print("\t\tvalidation dataset best MAE so far:\t{0:8.2f}\t\t(sMAPE: {1:6.2f} %)".
              format(trials.best_trial['result']['MAE Val'], trials.best_trial['result']['sMAPE Val']))
        print("\t\ttest dataset best MAE so far:\t\t{0:8.2f}\t\t(sMAPE: {1:6.2f} %)".
              format(trials.best_trial['result']['MAE Test'], trials.best_trial['result']['sMAPE Test']))
    else:
        print('\n\t~~~ there is no previous round\n\t\twe have no best error values so far')

    print('\n\t~~~ calibration period:\t[{0} -> {1}]'.format(calibration_start_at, df_train.index[-1]))
    print('\t~~~ test period:\t\t[{0} -> {1}]'.format(delivery_day_date, df_test.index[-1]))
    print("\n\t~~~ current hyperparameters from the feature space:")
    for key, value in hyperparameters.items():
        print("\t\t{0}: {1}".format(key, value))

    # if required according to hyperparameters, then the related datasets are scaled
    if hyperparameters.get('scaleX') in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [x_train, x_val, x_test], scaler_x = scaling(datasets=[x_train, x_val, x_test],
                                                     normalize=hyperparameters.get('scaleX'))

    else:
        scaler_x = None

    if hyperparameters.get('scaleY') in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [y_train, y_val, y_test], scaler_y = scaling(datasets=[y_train, y_val, y_test],
                                                     normalize=hyperparameters.get('scaleY'))

    else:
        scaler_y = None

    # compose a list of neuron's cardinality in each hidden layer, provided those are equal or greater than 50
    neurons = [int(hyperparameters.get('neurons{0}'.format(k + 1))) for k in range(n_layers)
               if int(hyperparameters.get('neurons{0}'.format(k + 1))) >= 50]

    # set seed according to the hyperparameter 'seed'
    np.random.seed(seed=int(hyperparameters.get('seed')))

    # instantiate a DNN model object based on keras and tensorflow using the parameters
    forecaster = DNNModel(neurons=neurons, n_features=x_train.shape[-1], output_shape=24,
                          dropout=hyperparameters.get('dropout'),
                          batch_normalization=hyperparameters.get('batch_normalization'),
                          lr=hyperparameters.get('lr'), verbose=False, scaler_x=scaler_x,
                          scaler_y=scaler_y, loss='mae', optimizer='adam', activation=hyperparameters.get('activation'),
                          initializer=hyperparameters.get('init'), regularization=hyperparameters.get('reg').get('val'),
                          lambda_reg=hyperparameters.get('reg').get('lambda'))

    # estimate the DNN model
    forecaster.seek_best_layer_weights(train_x=x_train, train_y=y_train, val_x=x_val, val_y=y_val, max_attempts=1000,
                                       epochs_early_stopping=20, batch_size=int(364//2))

    # make a prediction on the validation dataset
    # and remove axes of length one from the resulting array
    y_val_pred = forecaster.model.predict(x=x_val)

    # if the validation predictions are based on normalized values, then ...
    if hyperparameters.get('scaleY') in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        # denormalize those values back to their original scale
        # and remove axes of length one from the resulting array
        y_val = scaler_y.inverse_transform(dataset=y_val)
        y_val_pred = scaler_y.inverse_transform(dataset=y_val_pred)

    # calculate error metrics of the validation dataset
    mae_validation = np.mean(a=MAE(p_real=y_val, p_pred=y_val_pred))
    smape_validation = np.mean(a=sMAPE(p_real=y_val, p_pred=y_val_pred)) * 100

    # make a prediction on the test dataset
    # and remove axes of length one from the resulting array
    y_test_pred = forecaster.model.predict(x=x_test)

    # if the test predictions are based on normalized values, then ...
    if hyperparameters.get('scaleY') in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        # denormalize those values back to their original scale
        # and remove axes of length one from the resulting array
        y_test = scaler_y.inverse_transform(dataset=y_test)
        y_test_pred = scaler_y.inverse_transform(dataset=y_test_pred)

    # calculate error metrics of the test dataset
    mae_test = np.mean(a=MAE(p_real=y_test, p_pred=y_test_pred))
    smape_test = np.mean(a=sMAPE(p_real=y_test, p_pred=y_test_pred)) * 100

    # The test dataset is returned for directly evaluating the models without recalibration
    # while performing hyperopt. However, the hyperparameter search is performed using a validation
    # dataset
    return_values = {'loss': mae_validation, 'MAE Val': mae_validation, 'MAE Test': mae_test,
                     'sMAPE Val': smape_validation, 'sMAPE Test': smape_test, 'status': STATUS_OK}
    print('\t*** current error values:')
    print('\t\tvalidation dataset loss:\t{0:8.2f}'.format(return_values.get('loss')))
    print('\t\tvalidation dataset MAE:\t\t{0:8.2f}'.format(return_values.get('MAE Val')))
    print('\t\ttest dateset MAE:\t\t\t{0:8.2f}'.format(return_values.get('MAE Test')))
    print('\t\tvalidation dataset sMAPE:\t{0:8.2f} %'.format(return_values.get('sMAPE Val')))
    print('\t\ttest dateset sMAPE:\t\t\t{0:8.2f} %'.format(return_values.get('sMAPE Test')))
    print('\t\tstatus:\t\t\t\t\t\t\t\t{0}'.format(return_values.get('status')))
    print('   ', '*' * 80)
    print()

    return return_values


def hyperparameter_optimizer(path_datasets_folder=os.path.join('..', '..', 'examples', 'datasets'),
                             path_hyperparameter_folder=os.path.join('..', '..', 'examples', 'experimental_files'),
                             new_hyperopt=True, max_evals=1500, n_layers=2, dataset='DE', index_col=0, response_col=1,
                             sep=',', decimal='.', date_format='ISO8601', encoding='utf-8', index_tz='CET',
                             market_tz='CET', intended_freq='1h', price_lags=(1, 2, 3, 7), exog_lags=(0, 1, 7),
                             years_test=2, begin_test_date=None, end_test_date=None, calibration_window=364 * 4,
                             percentage_val=0.25, shuffle_train=True, data_augmentation=None, experiment_id=None,
                             summary=False):
    """ Function to optimize the hyperparameters and input features of the DNN.

    Parameters
    ----------
        path_datasets_folder : str
            Path to the folder where the input data are.

        path_hyperparameter_folder : str
            Path to a folder where the hyperopt's "trials" object will be put.

        new_hyperopt : bool
            Boolean that decides whether to start a new hyperparameter optimization or re-start an
            existing one.

        max_evals : int
            Maximum number of iterations for hyperparameter optimization.

        n_layers : int
            Number of layers of the DNN model.

        dataset : str
            Name of the dataset/market (as well as the csv format file) under study.
            If it is one of the standard markets, i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``,
            the dataset is automatically downloaded from the cloud.
            If the name is different, a dataset with a csv format must be placed into the ``path_datasets_folder``.

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
            Delimiter of the input csv file.
            In case the input dataset is one of the standard open-access benchmark ones,
            then you need not to set it.

        decimal : str
            Decimal point of numeric data in the input csv file.
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

        price_lags : tuple
            Daily lags of the response_col variable used for training and forecasting.
            The default values are (1, 2, 3, 7)

        exog_lags : tuple
            Daily lags of the exogenous variables used for training and forecasting.
            The default values are (0, 1, 7)

        years_test : int
            Number of years (a year is 364 days) in the test dataset.
            It is only used if the arguments ``begin_test_date`` and ``end_test_date`` are not provided.

        begin_test_date : datetime | str | None
            Optional parameter to select the test dataset.
            Used in combination with the argument ``end_test_date``.
            If either of them is not provided, the test dataset is built using the ``years_test`` argument.
            ``begin_test_date`` should either be a string with the following
            format ``"%Y-%m-%d 00:00:00"``, or a datetime object.

        end_test_date : datetime | str | None
            Optional parameter to select the test dataset.
            Used in combination with the argument ``begin_test_date``.
            If either of them is not provided, the test dataset is built using the ``years_test`` argument.
            ``end_test_date`` should either be a string with the following format ``"%Y-%m-%d 23:00:00"``,
            or a datetime object.

        calibration_window : int
            The number of days (a year is considered as 364 days)
            that are used in the training dataset for recalibration.
            Limits the training dataset starting point in a sliding window.

        percentage_val : float
            Percentage of data to be used for validation from the training dataset.

        shuffle_train : bool
            Boolean that selects whether the validation and training datasets
            are shuffled before splitting.
            Based on empirical results, this configuration does not play a role
            when selecting the hyperparameters.
            However, it is important when recalibrating the DNN model.

        data_augmentation : bool
            Boolean that selects whether a data augmentation technique for DNNs is used.
            Based on empirical results, for some markets data augmentation might
            improve forecasting accuracy at the expense of higher computational costs.
            Data augmentation technique enriches the training dataset with additional
            observations that are derived from the original ones.

        experiment_id : str | int |None
            A unique identifier which is added as a suffix to the 'trials' file name.
            If not provided, the current date is used as identifier.

        summary : bool
            Whether to print a summary of the resulting DataFrames.
    """
    # check if the provided directory for hyperparameter exists and if not, create it
    os.makedirs(name=path_hyperparameter_folder, exist_ok=True)

    # set experiment id
    if experiment_id is None:
        experiment_id = datetime.now().strftime(format="%Y%m%d_%H%M%S")
    else:
        experiment_id = experiment_id
    print("\n\t*** experiment ID: '{0}' ***".format(experiment_id))

    # define the 'trials' file name used to save the optimal hyperparameters
    trials_file_name = ('DNN_hyperparameters_nl{0}_dat{1}_YT{2}{3}{4}_CW{5}_{6}'.
                        format(str(n_layers), str(dataset), str(years_test),
                               '_SF' * int(shuffle_train),
                               '_DA' * int(0 if data_augmentation is None else data_augmentation),
                               str(calibration_window // 364), str(experiment_id)))

    # compose the full path of the 'trials' file
    trials_file_path = os.path.join(path_hyperparameter_folder, trials_file_name)
    print("\n\t~~~ Trials file path:", os.path.abspath(trials_file_path), sep="\n\t")

    # If hyperparameter optimization starts from scratch, then ...
    if new_hyperopt:
        # Instantiate a new database interface object (list of dictionaries)
        # supporting data-driven model-based optimization.
        # It is used by the hyperopt's fmin() function work by analyzing samples of a response surface.
        # It is a history of what points in the search space were tested, and what was discovered by those tests.
        # The 'trials' instance stores the history and makes it available to fmin().
        trials = Trials()
        print("\n\t~~~ new 'trials' object created")
    else:
        # Otherwise, an existing 'Trials' object is read from a priori saved pickle file.
        trials = pc.load(file=open(file=trials_file_path, mode="rb"))
        print("\n\t~~~ existing 'trials' object loaded from {0}".
              format(os.path.basename(trials_file_path)))
    print()

    # import the whole dataset
    df_all = read_data(path=path_datasets_folder, dataset=dataset, index_col=index_col, response_col=response_col,
                       sep=sep, decimal=decimal, date_format=date_format, encoding=encoding, index_tz=index_tz,
                       market_tz=market_tz, intended_freq=intended_freq, summary=summary)

    # split into train and test data
    df_train, df_test = split_data(df=df_all, years_test=years_test, begin_test_date=begin_test_date,
                                   end_test_date=end_test_date, summary=summary)

    # detect the exogenous input columns
    exogenous_inputs = [col for col in df_train.columns if 'exogenous' in col.casefold()]
    print("\n\t~~~ original exogenous explanatory variables:")
    for exog in exogenous_inputs:
        print("\t\t- '{0}'".format(exog))

    # calculate the number of exogenous input columns
    n_exogenous_inputs = len(exogenous_inputs)

    # Build hyperparameter search space. This includes hyperparameter and features.
    space = _build_space(n_layers=n_layers, data_augmentation=data_augmentation, n_exogenous_inputs=n_exogenous_inputs,
                         price_lags=price_lags, exog_lags=exog_lags)
    print("\n\t~~~ The hyperparameter/feature search space has generated.")
    print('*' * 100)

    # partially apply the given arguments and keywords to the prospective objective function
    fmin_objective = partial(_hyperopt_objective, trials=trials, trials_file_path=trials_file_path, max_evals=max_evals,
                             n_layers=n_layers, df_train=df_train, df_test=df_test,
                             calibration_window=calibration_window, percentage_val=percentage_val,
                             shuffle_train=shuffle_train)
    print("\n\t~~~ fmin_objective function has composed")

    # Perform an iterative hyperparameter optimization, which is minimizing the objective function over
    # a hyperparameter space.
    # More realistically: explore a function over a hyperparameter space according to a given algorithm,
    # allowing up to a certain number of function evaluations.
    # As points are explored, they are accumulated and saved into a 'trials' object and file.
    fmin(fn=fmin_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=False)


if __name__ == "__main__":
    # optimize the hyperparameters and input features of the DNN
    hyperparameter_optimizer(path_datasets_folder=os.path.join('..', '..', 'examples', 'datasets'),
                             path_hyperparameter_folder=os.path.join('..', '..', 'examples', 'experimental_files'),
                             new_hyperopt=True, max_evals=1500, n_layers=2, dataset='DE', index_col=0, response_col=1,
                             price_lags=(1, 2, 3, 7), exog_lags=(0, 1, 7), years_test=2, begin_test_date=None,
                             end_test_date=None, calibration_window=364 * 4, percentage_val=0.25, shuffle_train=True,
                             data_augmentation=True, experiment_id=1)
