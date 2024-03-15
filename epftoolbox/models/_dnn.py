"""
Classes and functions to implement the DNN model for electricity price forecasting.
This module does not include the hyperparameter optimization functions;
those are included in the _dnn_hyperopt module.
"""

# Author: Jesus Lago & Sandor Budai

# License: AGPL-3.0 License

import numpy as np
import pandas as pd
import pickle as pc
import os
import tensorflow.keras as kr

from datetime import datetime
from epftoolbox.evaluation import MAE, sMAPE
from epftoolbox.data import scaling, read_data, split_data
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, AlphaDropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.regularizers import l2, l1


class DNNModel(object):
    """ Basic DNN model based on keras and tensorflow.

    The model can be used standalone to train and predict a DNN using its fit/predict methods.
    However, it is intended to be used within the :class:`hyperparameter_optimizer` method
    and the :class:`DNN` class.
    The former obtains a set of the best hyperparameter using the :class:`DNNModel` class.
    The latter employees the set of best hyperparameters to recalibrate a :class:`DNNModel` object
    and make predictions.
    """

    def __init__(self, neurons, n_features, output_shape=24, dropout=0, batch_normalization=False, lr=None,
                 verbose=False, loss='mae', scaler_x=None, scaler_y=None,
                 optimizer='adam', activation='relu', initializer='glorot_uniform',
                 regularization=None, lambda_reg=0):
        """ Instantiate a DNNModel object.
        Parameters
        ----------
            neurons : list
                List containing the number of neurons in each hidden layer.
                E.g., if ``len(neurons)`` is 2, the DNN model has an input layer of size ``n_features``,
                two hidden layers, and an output layer of size ``outputShape``.

            n_features : int
                Number of input features in the model.
                This number defines the size of the input layer.

            output_shape : int
                Default number of output neurons.
                It is 24 as it is the default in most day-ahead markets.

            dropout : float
                Number between [0, 1] that selects the percentage of dropout.
                A value of 0 indicates no dropout.

            batch_normalization : bool
                Boolean that selects whether batch normalization is considered.

            lr : float
                Learning rate for optimizer algorithm.
                If none is provided, the default one is employed
                (see the `keras documentation <https://keras.io/>`_ for the default learning rates of each algorithm).

            verbose : bool
                Boolean that controls the logs.
                If set to true, a minimum amount of information is displayed.

            scaler_x : :class:`epftoolbox.data.DataScaler`
                Scaler object to invert-scale the input of the neural network if the neural network
                is trained with scaled inputs.

            scaler_y : :class:`epftoolbox.data.DataScaler`
                Scaler object to invert-scale the output of the neural network if the neural network
                is trained with scaled outputs.

            loss : str
                Loss to be used when training the neural network.
                Any of the regression losses defined in keras can be used.

            optimizer : str
                Name of the optimizer when training the DNN.
                Possible values are ``'adam'``, ``'RMSprop'``, ``'adagrad'`` or ``'adadelta'``.
                See the `keras documentation <https://keras.io/>`_ for a list of optimizers.

            activation : str
                Name of the activation function in the hidden layers.
                See the `keras documentation <https://keras.io/>`_ for a list of activation function.

            initializer : str
                Name of the initializer function for each weight of the neural network.
                See the `keras documentation <https://keras.io/>`_ for a list of initializer functions.

            regularization : str | None
                Name of the regularization technique.
                It can have three values ``'l2'`` for l2-norm regularization, ``'l1'`` for l1-norm regularization,
                or ``None`` for no regularization.

            lambda_reg : int
                The weight for regularization if ``regularization`` is ``'l2'`` or ``'l1'``.
        """
        # sanity checks
        if dropout > 1 or dropout < 0:
            raise ValueError('Dropout parameter must be between 0 and 1')

        if optimizer not in [None, 'adam', 'RMSprop', 'adagrad', 'adadelta']:
            raise ValueError('Optimizer "{0}" is not implemented, the optimizer must be one of the following: '
                             'adam, RMSprop, adagrad, adadelta'.format(optimizer))

        self.neurons = neurons
        self.n_features = n_features
        self.outputShape = output_shape
        self.dropout = dropout
        self.batch_normalization = batch_normalization
        self.verbose = verbose
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.loss = loss
        self.optimizer = optimizer
        self.activation = activation
        self.initializer = initializer
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        # build the DNN model structure based on the parameters
        self.model = self._build_model()

        # choose and instantiate the DNN model optimizer object
        opt = ({'adam': kr.optimizers.Adam(learning_rate=lr, clipvalue=10000),
                'RMSprop': kr.optimizers.RMSprop(learning_rate=lr, clipvalue=10000),
                'adagrad': kr.optimizers.Adagrad(learning_rate=lr, clipvalue=10000),
                'adadelta': kr.optimizers.Adadelta(learning_rate=lr, clipvalue=10000)}.
               get(self.optimizer, kr.optimizers.Adam(learning_rate=lr, clipvalue=10000)))

        # configure the DNN model for training
        self.model.compile(optimizer=opt, loss=self.loss)

    def _regulizer(self):
        """ Internal method to build a level1 or level2 regularizer for the DNN model,
        according to the on self.lambda_reg parameter value.

        Returns
        -------
            tensorflow.keras.regularizers.Regularizer | None
                The instantiated level1 or level2 regularizer object.
        """
        if self.regularization == 'l2':
            # A regularizer that applies a L2 regularization penalty.
            # The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))
            return l2(l2=self.lambda_reg)

        elif self.regularization == 'l1':
            # A regularizer that applies a L1 regularization penalty.
            # The L1 regularization penalty is computed as: loss = l1 * reduce_sum(abs(x))
            return l1(l1=self.lambda_reg)

        else:
            return None

    def _build_model(self):
        """ Internal method that defines the structure of the DNN model.

        Returns
        -------
            tensorflow.keras.models.Model
                A neural network model using keras and tensorflow
        """
        # Instantiate a Keras tensor.
        # A Keras tensor is a symbolic tensor-like object, which we augment with certain attributes
        # that allow us to build a Keras model just by knowing the inputs and outputs of the model.
        # For instance, if a, b and c are Keras tensors, it becomes possible to do:
        # model = Model(input=[a, b], output=c)
        input_shape = (None, self.n_features)
        past_data = Input(batch_shape=input_shape)
        past_dense = past_data

        if self.activation == 'selu':
            self.initializer = 'lecun_normal'

        for key, neurons in enumerate(self.neurons):
            if self.activation == 'LeakyReLU':
                past_dense = Dense(units=neurons, activation='linear', batch_input_shape=input_shape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._regulizer())(past_dense)
                past_dense = LeakyReLU(alpha=.001)(past_dense)

            elif self.activation == 'PReLU':
                past_dense = Dense(units=neurons, activation='linear', batch_input_shape=input_shape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._regulizer())(past_dense)
                past_dense = PReLU()(past_dense)

            else:
                past_dense = Dense(units=neurons, activation=self.activation, batch_input_shape=input_shape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._regulizer())(past_dense)

            if self.batch_normalization:
                # Apply a transformation that maintains the mean output close to 0
                # and the output standard deviation close to 1.
                past_dense = BatchNormalization()(past_dense)

            if self.dropout > 0:
                if self.activation == 'selu':
                    # Applies Alpha Dropout to the input, which keeps mean and variance of inputs to their original
                    # values.
                    # This can help to ensure the self-normalizing property even after this dropout.
                    # Alpha Dropout fits well to the scaled exponential linear units (selu) by randomly setting
                    # activations to the negative saturation value.
                    past_dense = AlphaDropout(self.dropout)(past_dense)
                else:
                    # Applies Dropout to the input, which randomly sets input units to 0 with a frequency of rate
                    # at each step during training time.
                    # This helps prevent over fitting.
                    # Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
                    past_dense = Dropout(self.dropout)(past_dense)

        # instantiate a regular densely connected NN layer object
        output_layer = Dense(units=self.outputShape,
                             kernel_initializer=self.initializer,
                             kernel_regularizer=self._regulizer())(past_dense)

        # instantiate a DNN model object grouping layers into in it with training/inference features.
        model = Model(inputs=[past_data], outputs=[output_layer])

        return model

    def _obtain_error_metrics(self, x_real, y_real):
        """ Internal method to update the metrics used to train the network.

        Parameters
        ----------
            x_real : numpy.ndarray
                A wide format array of explanatory variables for evaluating the DNN model.

            y_real : numpy.ndarray
                A wide format array of response variables for evaluating the DNN model.

        Returns
        -------
            list[float, float]
                The loss value and the MAE value of the DNN model on the provided dataset.
        """
        # calculate the LOSS of the DNN model in test mode based on the provided dataset
        loss = self.model.evaluate(x=x_real, y=y_real, verbose=0)

        # generate predictions based on the explanatory variables of the provided dataset
        y_pred = self.model.predict(x=x_real, verbose=0)

        # if the provided response dataset is scaled, then transformed back the predictions
        # and the observed values to their original scale
        if self.scaler_y is not None:
            if y_real.ndim == 1:
                y_real = y_real.reshape(-1, 1)
                y_pred = y_pred.reshape(-1, 1)

            y_real = self.scaler_y.inverse_transform(dataset=y_real)
            y_pred = self.scaler_y.inverse_transform(dataset=y_pred)

        # compute the mean absolute error (MAE) between predicted and observed values
        mae = MAE(p_real=y_real, p_pred=y_pred)

        return loss, mae

    def seek_best_layer_weights(self, train_x, train_y, val_x, val_y, max_attempts=1000, epochs_early_stopping=20,
                                batch_size=int(364 // 2)):
        """ Method to estimate the best layer weights of the DNN model based on several fitting and validation attempts.

        Parameters
        ----------
            train_x : numpy.ndarray
                Explanatory variables of the training dataset.

            train_y : numpy.ndarray
                Response variables of the training dataset.

            val_x : numpy.ndarray
                Explanatory variables of the validation dataset used for early-stopping.

            val_y : numpy.ndarray
                Response variables of the validation dataset used for early-stopping.

            max_attempts : int
                The maximum number of iterations to train and improve the DNN model.
                The default value is 1000.

            epochs_early_stopping : int
                Number of epochs used in early stopping to stop training.
                When no improvement is observed in the validation dataset after
                ``epochs_early_stopping`` epochs, the training stops.
                The default value is 20.

            batch_size : int
                The number of samples (rows)per gradient update.
                If unspecified, batch_size will default to 32.
                The default value is 364 // 2.
        """
        # for the first time set a really large value as the best error and best MAE
        best_error = 1e20
        best_mae = 1e20

        # for the first time retrieve the weights of the DNN model
        best_weights = self.model.get_weights()

        # preset an ancillary variable as zero for the first time to control
        # the number of training rounds without any improvement
        count_no_improvement = 0

        # iterate over the model training attempts
        print()
        for attempt in range(max_attempts):

            # Shuffle and train the DNN model in batch (using fix-sized bundles of records) on the training data sets.
            # Since epochs are set to 1, each batch is used only once.
            self.model.fit(x=train_x, y=train_y, batch_size=int(batch_size), epochs=1, verbose=False, shuffle=True)

            # calculate relevant error metrics (loss, MAE) of the DNN model based on the validation dataset
            val_error, val_mae = self._obtain_error_metrics(x_real=val_x, y_real=val_y)

            # Early-stopping
            # On the validation set check if any validation error metric is better (lower)
            # than the so far best one.
            if val_error < best_error or val_mae < best_mae:
                # retrieve the response variables' weights of the current model
                # and assign them to the best_weights object
                best_weights = self.model.get_weights()

                # set the current error metrics as the new best ones
                best_error = val_error
                best_mae = val_mae

                # reset the number of rounds without any improvement to 0
                count_no_improvement = 0

            else:
                # increase the number of rounds without any improvement by 1
                count_no_improvement += 1

            # display useful error information after each model training trial
            # no matter if it is better or worse than the so far best one
            if self.verbose:
                print("\tBest validation error so far:\t\t\t{0:.4f}".format(best_error))
                print("\tBest validation MAE so far:\t\t\t\t{0:.4f}".format(best_mae))
                print("\tSequence number of the attempt:\t\t\t{0}".format(attempt+1))
                print("\tNumber of attempts w/o improvements:\t{0}\n".format(count_no_improvement))

            # if the number of training rounds without improvement reaches the maximum allowed value
            # stop the training
            if count_no_improvement >= epochs_early_stopping:
                break

        # after several rounds of improving steps,
        # set the so far best weights as the weights of the layer into the model
        self.model.set_weights(weights=best_weights)


class DNN(object):
    """ Class to build a DNN model, recalibrate it, and use it to predict day-ahead electricity prices.

    It considers a set of the best hyperparameters, it recalibrates a :class:`DNNModel` based on
    these hyperparameters, and makes new predictions.

    The difference with respect to the :class:`DNNModel` class lies in the functionality.
    The :class:`DNNModel` class provides a simple interface to build a keras DNN model which
    is limited to fit and predict methods. This class extends the functionality by
    providing an interface to extract the best set of hyperparameters, and to perform recalibration
    before every prediction.

    Note that before using this class, a hyperparameter optimization run must be done using the
    :class:`hyperparameter_optimizer` function. Such hyperparameter optimization must be done
    using the same parameters: ``n_layers``, ``dataset``, ``years_test``, ``shuffle_train``,
    ``data_augmentation``, and ``calibration_window``

    An example on how to use this class is provided :ref:`here<dnnex3>`.
    """

    def __init__(self, experiment_id,
                 path_hyperparameter_file=None, n_layers=2, dataset='PJM', years_test=2, begin_test_date=None,
                 end_test_date=None, shuffle_train=1, data_augmentation=False, calibration_window=4 * 364,
                 scaler_x=None, scaler_y=None):
        """ Instantiate a DNN object.

        Parameters
        ----------
            experiment_id : str
                Unique identifier part of the 'trials' file name.
                In particular, every hyperparameter optimization set has a unique identifier associated with.
                See :class:`hyperparameter_optimizer` for further details.

            path_hyperparameter_file : str
                File path containing the 'trials' file with the optimal hyperparameters.

            n_layers : int
                Number of layers of the DNN model

            dataset : str
                The filename (w/o "csv" extension) of the input dataset/market under study.
                If one of the standard open-access benchmark datasets referred,
                such as "PJM", "NP", "BE", "FR", or "DE", then that will automatically be
                downloaded from zenodo.org into the ``path_datasets_folder``.
                In any other case, the input csv dataset should be placed in advance into the ``path_datasets_folder``.

            years_test : int
                Optional parameter to set the number of years of the being test dataset,
                counting back from the end of the input dataset.
                Note that a year is considered to be 364 days long.
                It is only used if the arguments `begin_test_date` and `end_test_date` are not provided.

            begin_test_date : datetime.datetime | str | None
                Optional parameter to select the test dataset.
                Used in combination with the argument `end_test_date`.
                If either of them is not provided, the test dataset will be split using the `years_test` argument.
                The `begin_test_date` should either be a string with the following format "%Y-%m-%d 00:00:00",
                or a datetime object.

            end_test_date : datetime.datetime | str | None
                Optional parameter to select the test dataset.
                This value may be earlier than the last timestamp in the input dataset.
                Used in combination with the argument `begin_test_date`.
                If either of them is not provided, the test dataset will be split using the `years_test` argument.
                The `end_test_date` should either be a string with the following format "%Y-%m-%d 23:00:00",
                or a datetime object.

            shuffle_train : bool
                Boolean that selects whether the validation and training datasets were shuffled when
                performing the hyperparameter optimization.
                Note that it does not select whether
                shuffling is used for recalibration as for recalibration the validation and the
                training datasets are always shuffled.

            data_augmentation : bool
                Boolean that selects whether a data augmentation technique for DNNs is used.
                Based on empirical results, for some markets data augmentation might
                improve forecasting accuracy at the expense of higher computational costs.
                Data augmentation technique enriches the training dataset with additional
                observations that are derived from the original ones.

            calibration_window : int
                Calibration window (in days) for the model training.
                Limits the training dataset starting point in a sliding window.
                The default value is 4 years * 364 days, namely 1456 days.

            scaler_x : :class:`epftoolbox.data.DataScaler`
                Scaler object to invert-scale the input of the neural network if the neural network
                is trained with scaled inputs.

            scaler_y : :class:`epftoolbox.data.DataScaler`
                Scaler object to invert-scale the output of the neural network if the neural network
                is trained with scaled outputs.
        """
        # sanity check of parameters
        if not isinstance(n_layers, int):
            raise Exception('n_layers must be an integer')

        if not isinstance(dataset, str):
            raise Exception('dataset must be a string')

        if begin_test_date is None and end_test_date is None and years_test is not None:
            if not isinstance(years_test, (float, int)):
                raise TypeError("the years_test should be a numerical object")

            if years_test <= 0:
                raise ValueError("The years_test should be positive!")

        elif begin_test_date is not None and end_test_date is not None:
            if not isinstance(begin_test_date, (pd.Timestamp, str)):
                raise TypeError("the begin_test_date should be a pandas Timestamp object"
                                " or a string with '%Y-%m-%d 00:00:00' format")

            if not isinstance(end_test_date, (pd.Timestamp, str)):
                raise TypeError("the end_test_date should be a pandas Timestamp object"
                                " or a string with '%Y-%m-%d 23:00:00' format")

        else:
            raise Exception("Please provide either the number of years for testing "
                            "or the start and end dates of the test period!")

        if not isinstance(shuffle_train, bool):
            raise Exception('shuffle_train must be a boolean')

        if not isinstance(data_augmentation, bool):
            raise Exception('data_augmentation must be a boolean')

        if not isinstance(calibration_window, int):
            raise Exception('calibration_window must be an integer')

        if not os.path.exists(path_hyperparameter_file):
            raise Exception('the provided hyperparameter file does not exist')

        # set Unique identifier part of the 'trials' file name
        self.experiment_id = experiment_id

        # set the folder path of the 'trials' file with the optimal hyperparameters
        self.path_hyperparameter_file = path_hyperparameter_file

        # set the number of layers in the prospecting DNN model
        self.n_layers = n_layers

        # set the filename (w/o "csv" extension) of the input dataset/market under study.
        self.dataset = dataset

        # set length of number of years (a year is 364 days) of the test dataset
        self.years_test = years_test
        self.begin_test_date = begin_test_date
        self.end_test_date = end_test_date

        # set the validation and training datasets were shuffled when
        # performing the hyperparameter optimization
        self.shuffle_train = shuffle_train

        # set whether data augmentation technique is employed
        self.data_augmentation = data_augmentation

        # set the calibration window, number of days
        self.calibration_window = calibration_window

        # set x and y scalers
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        # set the a priori non defined parameters as None
        self.trained = None
        self.delivery_day_date = None

        # read and set the best hyperparameters from the corresponding pickle file
        self._read_best_hyperapameters()

    def _read_best_hyperapameters(self):
        """ Internal method, which reads the 'trials' file, extracts optimal hyperparameters,
        convert them to a shallow python dictionary and saves that dictionary in the class.

        The file that is read depends on the provided input parameters to the class.
        """
        # read the pickled representation of the optimal hyperparameters and extract the best hyperparameters
        # TODO: further improvement possibility
        trials = pc.load(file=open(file=self.path_hyperparameter_file, mode="rb"))
        best_trial = trials.best_trial

        # convert the best trial to a dictionary and save it in the class
        self.best_hyperparameters = self._format_best_trial(best_trial=best_trial)
        print("\n\tThe hyperparameters of the best trial have been extracted from '{0}' pickle file.".
              format(os.path.abspath(self.path_hyperparameter_file)))
        for key, val in self.best_hyperparameters.items():
            print("\t\t'{0}' = {1}".format(key, val))

    @staticmethod
    def _format_best_trial(best_trial):
        """ Function to format a ``trials.best_trials`` object to a shallow python dictionary along with some value
        translation.

        Parameters
        ----------
            best_trial : dict
                A not shallow dictionary as extracted from the ``trials.best_trials`` object,
                which is generated by the :class:`hyperparameter_optimizer`function.

        Returns
        -------
            dict
                Formatted and translated dictionary containing the optimal hyperparameters.
        """
        # extract the hyperparameters of the best trial
        # and unpack the list format of the dictionary values
        best_hyperparameters = best_trial.get('misc').get('vals')
        best_hyperparameters = {key: 0 if len(val) == 0 else val[0] for key, val in best_hyperparameters.items()}

        # translate discrete hyperparameters from integer value to string representation
        translation_dict = {'activation': ["relu", "softplus", "tanh", 'selu', 'LeakyReLU', 'PReLU', 'sigmoid'],
                            'init': ['Orthogonal', 'lecun_uniform', 'glorot_uniform', 'glorot_normal', 'he_uniform',
                                     'he_normal'],
                            'scaleX': ['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant'],
                            'scaleY': ['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant'],
                            'reg': [None, 'l1']}
        for trans_key in ['activation', 'init', 'scaleX', 'scaleY', 'reg']:
            best_hyperparameters[trans_key] = translation_dict.get(trans_key)[best_hyperparameters.get(trans_key)]

        return best_hyperparameters

    def fit(self, x_train, y_train, x_val, y_val, optimizer, loss, verbose):
        """ Method that instantiate a :class:`DNNModel` object and fit it.

        The method trains several :class:`DNNModel` model objects and validates them on the validation dataset,
        using the set of optimal hyperparameters that are found in ``path_hyperparameter_file``.

        Parameters
        ----------
            x_train : numpy.ndarray
                Input of the training dataset

            x_val : numpy.ndarray
                Input of the validation dataset

            y_train : numpy.ndarray
                Output of the training dataset

            y_val : numpy.ndarray
                Output of the validation dataset

            optimizer : str
                Name of the optimizer when training the DNN.
                Possible values are ``'adam'``, ``'RMSprop'``, ``'adagrad'`` or ``'adadelta'``.
                See the `keras documentation <https://keras.io/>`_ for a list of optimizers.

            loss : str
                Loss to be used when training the neural network.
                Any of the regression losses defined in keras can be used.

            verbose : bool
                Boolean that controls the logs.
                If set to true, a minimum amount of information is displayed.
        """
        # compose a list of neuron's cardinality in each hidden layer, provided those are equal or greater than 50
        neurons = [int(self.best_hyperparameters.get('neurons{0}'.format(k + 1))) for k in range(self.n_layers)
                   if int(self.best_hyperparameters.get('neurons{0}'.format(k + 1))) >= 50]

        # set seed according to the hyperparameter 'seed'
        np.random.seed(seed=int(self.best_hyperparameters.get('seed')))

        # instantiate a DNN model object based on keras and tensorflow using the best hyperparameters
        self.trained = DNNModel(neurons=neurons, n_features=x_train.shape[-1],
                                dropout=self.best_hyperparameters.get('dropout'),
                                batch_normalization=self.best_hyperparameters.get('batch_normalization'),
                                lr=self.best_hyperparameters.get('lr'), verbose=verbose, optimizer=optimizer,
                                activation=self.best_hyperparameters.get('activation'), loss=loss,
                                scaler_x=self.scaler_x, scaler_y=self.scaler_y,
                                regularization=self.best_hyperparameters.get('reg'),
                                lambda_reg=self.best_hyperparameters.get('lambdal1'),
                                initializer=self.best_hyperparameters.get('init'))

        # experiment the best layer weights of the DNN model based on several fitting and validation attempts
        self.trained.seek_best_layer_weights(train_x=x_train, train_y=y_train, val_x=x_val, val_y=y_val,
                                             max_attempts=1000, epochs_early_stopping=20, batch_size=int(364//2))

    def predict(self, x_test):
        """ Method that makes a prediction using some given inputs

        Parameters
        ----------
            x_test : numpy.ndarray
                Explanatory variables of the test period.

        Returns
        -------
            numpy.ndarray
                An array containing the day-ahead electricity price predictions
                for each product of the in scope day(s).
        """
        # Predict the current date using a recalibrated DNN
        # and remove axes of length one from the resulting array
        y_pred = self.trained.model.predict(x=x_test)
        y_pred = y_pred.squeeze().reshape(1, -1)

        # Inverse transforms the predictions
        if self.best_hyperparameters.get('scaleY') in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            y_pred = self.scaler_y.inverse_transform(dataset=y_pred)

        return y_pred

    def transform_split_recalibrate_fit_forecast(self, df, shuffle_train=True, percentage_val=0.25,
                                                 hyperoptimization=False, data_augmentation=False,
                                                 delivery_date_start=None, optimizer='adam', loss='mae', verbose=False):
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
                of electricity delivery periods.
                The column names should follow this convention:
                ``['Price', 'Exogenous_1', 'Exogenous_2', ..., 'Exogenous_N']``.

            shuffle_train : bool
                If true, the validation and training datasets are shuffled.

            percentage_val : float
                Percentage of data to be used for validation from the training dataset.

            hyperoptimization : bool
                Fix (or not) the random shuffle index so that the validation dataset does not change
                during the hyperparameter optimization process.

            data_augmentation : bool
                Boolean that selects whether a data augmentation technique for recalibration is used.
                Based on empirical results, for some markets data augmentation might
                improve forecasting accuracy at the expense of higher computational costs.
                Data augmentation technique enriches the training dataset with additional
                observations that are derived from the original ones.

            delivery_date_start : datetime.datetime
                Date of the electricity delivery day.

            optimizer : str
                Name of the optimizer when training the DNN.
                Possible values are ``'adam'``, ``'RMSprop'``, ``'adagrad'`` or ``'adadelta'``.
                See the `keras documentation <https://keras.io/>`_ for a list of optimizers.

            loss : str
                Loss to be used when training the neural network.
                Any of the regression losses defined in keras can be used.

            verbose : bool
                Boolean that controls the logs.
                If set to true, a minimum amount of information is displayed.

        Returns
        -------
            numpy.ndarray
                The prediction of day-ahead prices.
        """
        # extract the starting datetime of the in-focus delivery day
        # self.delivery_day_date = df.loc[df.Price.isnull()].index[0]
        self.delivery_day_date = delivery_date_start

        # calculate the starting datetime of the calibration window
        calibration_start_at = (delivery_date_start - pd.Timedelta(days=self.calibration_window))

        # turn the dataframe into a wide format (from hourly to daily)
        # and add the lags of variables as new columns
        # and add day of week dummies if requested
        x_train, y_train, x_val, y_val, x_test, _ = pivot_lag_extend(
            df_all=df, features=self.best_hyperparameters, data_augmentation=data_augmentation,
            calibration_start_at=calibration_start_at, begin_test_date=self.delivery_day_date,
            end_test_date=df.index[-1], percentage_val=percentage_val, shuffle_train=shuffle_train,
            hyperoptimization=hyperoptimization)

        # scale datasets according to the corresponding parameters of the best hyperparameter set, if required
        if self.best_hyperparameters.get('scaleX') in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            [x_train, x_val, x_test], self.scaler_x = scaling(datasets=[x_train, x_val, x_test],
                                                              normalize=self.best_hyperparameters.get('scaleX'))
        else:
            self.scaler_x = None

        # scale datasets according to the corresponding parameters of the best hyperparameter set, if required
        if self.best_hyperparameters.get('scaleY') in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            [y_train, y_val], self.scaler_y = scaling(datasets=[y_train, y_val],
                                                      normalize=self.best_hyperparameters.get('scaleY'))
        else:
            self.scaler_y = None

        # Recalibrate the neural network and extracting the prediction
        self.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, optimizer=optimizer, loss=loss,
                 verbose=verbose)

        # make predictions based on the explanatory variables of the test dataset
        y_pred = self.predict(x_test=x_test)

        # Resets all state generated by Keras.
        kr.backend.clear_session()

        return y_pred


def pivot_lag_extend(df_all, features, data_augmentation, calibration_start_at, begin_test_date, end_test_date,
                     years_test=None, percentage_val=0.25, shuffle_train=True, hyperoptimization=False):
    """ Internal function to turn the long dataframe wider (from hourly to daily resolution),
    to calculate lagged values of variables as new columns,
    and to add day of week dummies as new columns.

    Parameters
    ----------
        df_all : pandas.DataFrame
            A 'long' dataframe containing the electricity day-ahead prices and some exogenous variables
            in hourly resolution.

        features : dict
            Dictionary that define the selected input features.
            The dictionary is based on the results of a hyperparameter/feature optimization run
            using the :class:`hyperparameter_optimizer`function

        data_augmentation : bool
            Boolean that selects whether a data augmentation technique for DNNs is used.
            Based on empirical results, for some markets data augmentation might
            improve forecasting accuracy at the expense of higher computational costs.
            Data augmentation technique enriches the training dataset with additional
            observations that are derived from the original ones.

        calibration_start_at : pandas.Timestamp
            The date of the first delivery.

        begin_test_date : datetime | str
            Optional parameter to select the test dataset.
            Used in combination with the argument
            end_test_date.
            If either of them is not provided, the test dataset is built using the
            years_test argument.
            The begin_test_date should either be a string with the following
            format d/m/Y H:M, or a datetime object.

        end_test_date : datetime | str
            Optional parameter to select the test dataset
            Used in combination with the argument begin_test_date.
            If either of them is not provided, the test dataset is built using the
            years_test argument.
            The end_test_date should either be a string with the following
            format d/m/Y H:M, or a datetime object.

        years_test : int
            Optional parameter to set the number of years of the being test dataset,
            counting back from the end of the input dataset.
            Note that a year is considered to be 364 days long.
            It is only used if the arguments `begin_test_date` and `end_test_date` are not provided.

        percentage_val : float
            Percentage of data to be used for validation from the training dataset.

        shuffle_train : bool
            Boolean that selects whether the validation and training datasets were shuffled when
            performing the hyperparameter optimization.
            Note that it does not select whether shuffling is used for recalibration
            as for recalibration the validation and the training datasets are always shuffled.

        hyperoptimization : bool
            Fix (or not) the random shuffle index so that the validation dataset does not change
            during the hyperparameter optimization process.

    Returns
    -------
        list[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            A wide dataframe containing the original and lagged day-ahead electricity prices
            and the original and lagged exogenous variables in daily resolution.
            Each hours' value is put in different columns.
    """
    # check whether the begin_test_date and end_test_date or years_test arguments are provided
    if (begin_test_date is None or end_test_date is None) and years_test is None:
        raise Exception('Either the argument `years_test` or `begin_test_date` and `end_test_date` should be provided.')
    if begin_test_date is None and end_test_date is None and years_test is not None:
        begin_test_date = df_all.index[-1] - pd.Timedelta(days=years_test * 364)
        end_test_date = df_all.index[-1]

    # check whether the last index in the dataframe is before the calibration start date
    if df_all.index[-1] < calibration_start_at:
        raise Exception('The last index in the dataframe is before the calibration start date.')

    # check whether the first index in the dataframe is after the first delivery day
    if df_all.index[0] > begin_test_date:
        raise Exception('The first index in the dataframe is after the first delivery day.')

    # check whether the calibration start date is after the first delivery day
    if calibration_start_at >= begin_test_date:
        raise Exception('The calibration start date is after the first delivery day.')

    # check whether the calibration_start_at corresponds with the hour 00:00
    if calibration_start_at.hour != 0:
        raise Exception('The first delivery day does not correspond with the hour 00:00.')

    # check whether the first_delivery_day corresponds with the hour 00:00
    if begin_test_date.hour != 0:
        raise Exception('The first delivery day does not correspond with the hour 00:00.')

    # create a copy of the inpt dataframe and adjust the column names for later processing
    first_frame = df_all.copy()
    first_frame.columns = [col + "_h{0}" for col in first_frame.columns.to_list()]

    # create a list which first element is a dataframe of day-ahead electricity prices
    shifted_responses = [first_frame]

    # add the day of the week (and part of the day) as a numeric feature, if needed
    if features.get('In: Day'):
        # Each value is the numeric 'day of the week' value plus the 'hour of day divided by 24' value.
        # So monday at 00 is 1. Monday at 1h is 1.0417, Tuesday at 1h is 2.0417, etc.
        daytime_df = pd.DataFrame(data=(df_all.index.dayofweek + df_all.index.hour / 24).to_list(),
                                  index=df_all.index,
                                  columns=['day_of_week'])
        shifted_responses.append(daytime_df)

    # extract the valid price lags from the feature dictionary
    valid_price_features = {key: value for key, value in features.items()
                            if key.startswith('In: Price ') and value != 0}

    # if there are 'Price' related keys in the feature dictionary
    if len(valid_price_features):
        # extract the price lags from the feature dictionary
        valid_price_lags = [key.split('-')[-1].split(' ')[-1] for key in valid_price_features.keys()]

        # Iterate over the needed price_lags to calculate accordingly shifted response_col variables
        # for each datetime index.
        # Each lagged value composes a new one-column temporary dataframe.
        for past_day in valid_price_lags:
            shifted_df = pd.DataFrame(data=df_all.Price.values,
                                      index=df_all.index + pd.Timedelta(days=int(past_day)),
                                      columns=['Price_h{0}_d' + past_day])
            shifted_responses.append(shifted_df)

    # extract the exogenous variables and their related valid lags from the feature dictionary
    valid_exog_features = [key.replace('In: Exog-', '') for key, value in features.items()
                           if key.startswith('In: Exog-') and value != 0]
    valid_exog_features = [(elem.split(' ')[0], elem.split(' ')[-1].replace('D', '')) for elem in valid_exog_features]
    valid_exog_features = [(elem[0], abs(int(elem[-1]))) for elem in valid_exog_features if not elem[-1] == '']

    # extract the sequence numbers of those exogenous variables,
    # which are needed according to the actual feature dictionary.
    sequence_nr_exog = sorted(list(set([elem[0]for elem in valid_exog_features])))

    # iterate over each valid exogenous input from the 'features' dictionary
    for i in sequence_nr_exog:
        # extracts the valid exog lags from the valid_exog_features list
        valid_exog_lags = [elem[-1] for elem in valid_exog_features if elem[0] == i]

        # if there are valid_exog_lags
        if valid_exog_lags:
            # compose the original exogenous variable name
            exog = "Exogenous_{0}".format(i)

            # Iterate over the needed exog_lags to calculate accordingly shifted exogenous variables
            # for each datetime index.
            # Each lagged value composes a new one-column temporary dataframe.
            for past_day in valid_exog_lags:
                shifted_df = pd.DataFrame(data=df_all[exog].values,
                                          index=df_all.index + pd.Timedelta(days=int(past_day)),
                                          columns=[exog + '_h{0}_d' + str(abs(past_day))])
                shifted_responses.append(shifted_df)

    # Add column-wise up the new lagged values to form a new lagged dataframe
    # of response and explanatory variables for each datetime index.
    df_lagged = pd.concat(objs=shifted_responses, axis=1, join='inner')

    # detect the inferred frequency of the ind
    long_inferred_freq = df_lagged.index.inferred_freq

    # create a list of those column names which should be pivoted wider
    pivot_wider_column_list = df_lagged.columns.to_list().copy()

    # if the 'In: Day' feature is present, then ...
    if features.get('In: Day'):
        # remove the 'day_of_week' column from the pivot_wider_column_list
        pivot_wider_column_list.remove('day_of_week')

        # move the day_of_week column to the first position
        cols = ['day_of_week'] + pivot_wider_column_list
        df_lagged = df_lagged[cols]

    # calculate daily delivery period numbers
    daily_delivery_period_numbers = {'h': 24, '30min': 48, '15min': 96, '5min': 288}. \
        get(long_inferred_freq, 24)

    # create an empty list to store the pivoted columns
    pivoted_columns = []

    # if the 'In: Day' feature is present, then ...
    if features.get('In: Day'):
        # append the 'day_of_week' column as the first element of the pivoted column list
        pivoted_columns.append(df_lagged.loc[:, ['day_of_week']])

    # iterate over the pivot_wider_column_list
    for col in pivot_wider_column_list:
        # iterate over the daily delivery period numbers
        for period in range(daily_delivery_period_numbers):
            # calculate the period lagged values of the column
            pivoted_column = df_lagged.loc[:, [col]].shift(periods=-1 * period)
            # rename the column
            pivoted_column.rename(columns={name: name.format(str(period).zfill(2)) for name
                                           in pivoted_column.columns.to_list()},
                                  inplace=True)
            # append the lagged values to the shifted_columns list
            pivoted_columns.append(pivoted_column)

    # append the shifted columns into a wide dataframe
    df_lagged_wide = pd.concat(objs=pivoted_columns, axis=1, join='inner').dropna(inplace=False)

    # drop rows with missing values
    df_lagged_wide.dropna(inplace=True)

    # adjust column names
    df_lagged_wide.columns.name = 'variable'

    # split the train data timewise
    df_train = df_lagged_wide.loc[(df_lagged_wide.index >= calibration_start_at) &
                                  (df_lagged_wide.index < begin_test_date)]

    # if data_augmentation is not True, then ...
    if not data_augmentation:
        # keep only the hourly train data
        df_train = df_train.loc[df_train.index.hour == 0]

    # split the test data timewise
    df_test = df_lagged_wide.loc[(df_lagged_wide.index >= begin_test_date) &
                                 (df_lagged_wide.index < end_test_date)]

    # keep only the hourly test data
    df_test = df_test.loc[df_test.index.hour == 0]

    # split the train data into X and Y
    x_train = df_train.filter(regex=r'^(?!^Price_h[0-9]{2}$).*', axis=1)
    y_train = df_train.filter(regex=r'^Price_h[0-9]{2}$', axis=1)

    # split the test data into X and Y
    x_test = df_test.filter(regex=r'^(?!^Price_h[0-9]{2}$).*', axis=1)
    y_test = df_test.filter(regex=r'^Price_h[0-9]{2}$', axis=1)

    print("\tcalibration_start_at", calibration_start_at)
    print("\tbegin_test_date - end_test_date: {0} - {1}".format(begin_test_date, end_test_date))

    # convert dataframes to numpy arrays
    x_train, y_train, x_test, y_test = x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy()

    # calculate the size of the validation dataset (number of rows)
    n_val = int(percentage_val * x_train.shape[0])

    # shuffle training dataset before further splitting, if needed
    if shuffle_train:

        if hyperoptimization:
            # We fixed the random shuffle index so that the validation dataset does not change
            # during the hyperparameter optimization process
            np.random.seed(seed=7)

        # We shuffle the data per week to avoid data contamination
        row_ind = np.arange(x_train.shape[0])
        row_ind_week = row_ind[::7]
        np.random.shuffle(x=row_ind_week)
        shuffled_ind = [ind + i for ind in row_ind_week for i in range(7) if ind + i in row_ind]

        x_train = x_train[shuffled_ind]
        y_train = y_train[shuffled_ind]

    # further split the training dataset into training and validation datasets
    n_train = x_train.shape[0] - n_val  # complements n_val
    x_val = x_train[n_train:]  # last n_val obs
    x_train = x_train[:n_train]  # first n_train obs
    y_val = y_train[n_train:]
    y_train = y_train[:n_train]

    return [x_train, y_train, x_val, y_val, x_test, y_test]


def evaluate_dnn_in_test_dataset(experiment_id, path_datasets_folder=os.path.join('..', '..', 'examples', 'datasets'),
                                 path_hyperparameter_file=None,
                                 path_recalibration_folder=os.path.join('..', '..', 'examples', 'experimental_files'),
                                 n_layers=2, dataset='PJM', years_test=2,
                                 begin_test_date=None, end_test_date=None,
                                 shuffle_train=True, percentage_val=0.25,
                                 data_augmentation=False, calibration_window=4, new_recalibration=False,
                                 index_col=None, response_col=None, sep=',', decimal='.', date_format='ISO8601',
                                 encoding='utf-8', save_frequency=10, index_tz=None, market_tz=None, intended_freq='1h',
                                 summary=False, optimizer='adam', loss='mae', verbose=False):
    """ Easy evaluation of the DNN model in a test dataset using daily recalibration.

    The test dataset is defined by a market name and the test dates.
    The function generates the test and training datasets, and evaluates a DNN model considering daily recalibration
    and an optimal set of hyperparameters.

    Note that before using this class, a hyperparameter optimization run must be done using the
    :class:`hyperparameter_optimizer` function. Moreover, the hyperparameter optimization must be done
    using the same parameters: ``n_layers``, ``dataset``, ``shuffle_train``,
    ``data_augmentation``, ``calibration_window``, and either the ``years_test`` or the same
    ``begin_test_date``/``end_test_date``

    An example on how to use this function is provided :ref:`here<dnnex2>`.

    Parameters
    ----------
        experiment_id : str
            Unique identifier to read the 'trials' file. In particular, every hyperparameter optimization
            set has a unique identifier associated with. See :class:`hyperparameter_optimizer` for further
            details

        path_datasets_folder : str
            Path where the datasets are stored or, if they do not exist yet, the path where the datasets
            are to be stored

        path_hyperparameter_file : str
            Path of the file containing the 'trials' file with the optimal hyperparameters

        path_recalibration_folder : str
            Path to save the forecast of the test dataset

        n_layers : int
            Number of hidden layers in the neural network

        dataset : str
            Name of the dataset/market under study. If it is one of the standard markets,
            i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded.
            If the name is different, a dataset with a csv format should be placed in the ``path_datasets_folder``.

        years_test : int
            Optional parameter to set the number of years of the being test dataset,
            counting back from the end of the input dataset.
            Note that a year is considered to be 364 days long.
            It is only used if the arguments `begin_test_date` and `end_test_date` are not provided.

        begin_test_date : datetime | str
            Optional parameter to select the test dataset. Used in combination with the argument
            end_test_date. If either of them is not provided, the test dataset is built using the
            years_test argument. The begin_test_date should either be a string with the following
            format d/m/Y H:M, or a datetime object.

        end_test_date : datetime | str
            Optional parameter to select the test dataset. Used in combination with the argument
            begin_test_date. If either of them is not provided, the test dataset is built using the
            years_test argument. The end_test_date should either be a string with the following
            format d/m/Y H:M, or a datetime object.

        shuffle_train : bool
            Boolean that selects whether the validation and training datasets were shuffled when
            performing the hyperparameter optimization. Note that it does not select whether
            shuffling is used for recalibration as for recalibration the validation and the
            training datasets are always shuffled.

        percentage_val : float
            Percentage of data to be used for validation from the training dataset.

        data_augmentation : bool
            Boolean that selects whether a data augmentation technique for DNNs is used.
            Based on empirical results, for some markets data augmentation might
            improve forecasting accuracy at the expense of higher computational costs.
            Data augmentation technique enriches the training dataset with additional
            observations that are derived from the original ones.

        calibration_window : int
            Number of days used in the training/validation dataset for recalibration.
            Limits the training dataset starting point in a sliding window.

        new_recalibration : bool
            Boolean that selects whether a new recalibration is performed or the function re-starts an old one.
            To restart an old one, the .csv file with the forecast must exist in the
            ``path_recalibration_folder`` folder

        index_col : int | str | None
            Column name of the index in the dataset

        response_col : int | str | None
            Column name of the response variable in the dataset

        sep : str
            Column separator character in the dataset

        decimal : str
            Decimal separator character in the dataset

        date_format : str
            Date format in the dataset

        encoding: str
            Encoding of the dataset

        index_tz : str
            Time zone of the index in the dataset

        market_tz : str
            Time zone of the market in the dataset

        intended_freq : str
            Frequency of the dataset

        save_frequency : int
            Frequency with which the DNN model is saved in the folder ``path_recalibration_folder``

        summary : bool
            Boolean that selects whether the summary of the DNN model is printed.

        optimizer : str
            Name of the optimizer when training the DNN. Possible values are ``'adam'``,
            ``'RMSprop'``, ``'adagrad'`` or ``'adadelta'``.
            See the `keras documentation <https://keras.io/>`_ for a list of optimizers.

        loss : str
            Loss to be used when training the neural network.
            Any of the regression losses defined in keras can be used.

        verbose : bool
            Boolean that controls the logs.
            If set to true, a minimum amount of information is displayed.

    Returns
    -------
        pandas.DataFrame
            A dataframe with all the predictions in the test dataset. The dataframe is also
            written to the folder ``path_recalibration_folder``
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
    daily_delivery_period_numbers = {'h': 24, '30min': 48, '15min': 96, '5min': 288}. \
        get(real_values.index.inferred_freq, 24)

    # define unique name to save the forecast
    forecast_file_name = ('DNN_forecast_nl{0}_dat{1}_YT{2}_SFH{3}{4}_CW{5}_{6}{7}.csv'.
                          format(str(n_layers), str(dataset), str(years_test), str(shuffle_train),
                                 '_DA' * int(data_augmentation), str(calibration_window), str(experiment_id),
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

    print('\n\t*** data_augmentation ***', data_augmentation, sep='\n\t')
    print('\n\t*** new_recalibration ***', new_recalibration, sep='\n\t')

    # Check whether start a new model recalibration or re-start an old one
    if new_recalibration:
        forecast_dates = forecast.index
        print('\n\t*** forecast_dates ***')
        print('\tfrom:', forecast_dates.to_list()[0])
        print('\tto:', forecast_dates.to_list()[-1])
        print('\tinferred_freq:', forecast.index.inferred_freq)

    else:
        # import the already existing forecast file
        forecast = pd.read_csv(filepath_or_buffer=forecast_file_path, index_col=0)
        forecast.index = pd.to_datetime(forecast.index)

        # read the dates to still be forecasted by checking NaN values
        forecast_dates = forecast[forecast.isna().any(axis=1)].index

        # If all the dates to be forecasted have already been forecast,
        if len(forecast_dates) == 0:

            # we print prediction metrics information and exit the script
            mae = np.mean(a=MAE(p_real=real_values.values, p_pred=forecast.values.squeeze()))
            smape = np.mean(a=sMAPE(p_real=real_values.values, p_pred=forecast.values.squeeze())) * 100
            print('\n\t*** forecast_dates ***')
            print('\tfrom:', forecast_dates.to_list()[0])
            print('\tto:', forecast_dates.to_list()[-1])
            print('\tinferred_freq:', forecast.index.inferred_freq)
            print('\tFinal metrics - sMAPE: {0:.2f}%  |  MAE: {1:.3f}'.format(smape, mae))
            return

    # instantiate a LEAR model object with the given calibration window
    model = DNN(experiment_id=experiment_id, path_hyperparameter_file=path_hyperparameter_file,
                n_layers=n_layers, dataset=dataset, years_test=years_test, begin_test_date=begin_test_date,
                end_test_date=end_test_date, shuffle_train=shuffle_train, data_augmentation=data_augmentation,
                calibration_window=calibration_window)

    # For loop over the recalibration dates
    for key, delivery_date_start in enumerate(forecast.index):

        # calculate the last timestamp of the in-focus delivery day
        # this is the last one before the next delivery day's start
        delivery_date_end = df_all.index[df_all.index < delivery_date_start + pd.Timedelta(days=1)][-1]
        print("\n\tinferred_freq of the whole data:", df_all.index.inferred_freq)

        # slice a new (long) dataframe from the whole data up to the in-focus delivery day's end
        data_available = df_all.loc[:delivery_date_end].copy()

        # Recalibrate the model with the most up-to-date available data
        # and making a prediction for the current_date
        y_pred = model.transform_split_recalibrate_fit_forecast(df=data_available, shuffle_train=shuffle_train,
                                                                percentage_val=percentage_val, hyperoptimization=False,
                                                                data_augmentation=data_augmentation,
                                                                delivery_date_start=delivery_date_start,
                                                                optimizer=optimizer, loss=loss, verbose=verbose)

        # fill up the forecast (wide) dataframe with the current_date's predictions
        forecast.loc[delivery_date_start, :] = y_pred

        # compute metrics up to current_date
        mae = np.mean(a=MAE(p_real=real_values.loc[:delivery_date_start].values.squeeze(),
                            p_pred=forecast.loc[:delivery_date_start].values.squeeze()))
        smape = np.mean(a=sMAPE(p_real=real_values.loc[:delivery_date_start].values.squeeze(),
                                p_pred=forecast.loc[:delivery_date_start].values.squeeze())) * 100

        # print error information
        print('\n\t{0} - sMAPE: {1:.2f}%  |  MAE: {2:.3f}'.format(str(delivery_date_start)[:10], smape, mae))

        # save the forecasts in save_frequency days chunks
        if (key + 1) % save_frequency == 0 or (key + 1) == len(forecast.index):
            forecast.to_csv(path_or_buf=forecast_file_path, sep=sep, decimal=decimal, date_format='%Y-%m-%d',
                            encoding=encoding, index=True, index_label='date')

    return forecast


if __name__ == '__main__':
    # These codes below will only be executed if this script is run as the main program.
    predictions = evaluate_dnn_in_test_dataset(
        experiment_id='1', path_hyperparameter_file=os.path.join('..', '..', 'examples', 'experimental_files',
                                                                 'DNN_hyperparameters_nl2_datDE_YT2_SF_CW4_1'),
        n_layers=2, dataset='DE', data_augmentation=False, calibration_window=364 * 4, new_recalibration=True,
        index_col=0, response_col=1, index_tz='CET', market_tz='CET',  # years_test=2,
        begin_test_date='2017-12-01 00:00:00', end_test_date='2017-12-10 23:00', shuffle_train=True,
        percentage_val=0.25, summary=True, optimizer='adam', loss='mae', verbose=False
    )
    print("\n\n\t*** predictions ***", predictions, sep='\n')
