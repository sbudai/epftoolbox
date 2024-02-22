"""
Classes and functions to implement data wrangling operations. At the moment, this is limited to
data scaling.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.robust import mad


class MedianScaler(object):
    """ A class to normalize and denormalize data using median and robust median absolute deviation values. """

    def __init__(self):
        self.median = None
        self.mad = None
        self.fitted = False

    def fit(self, data) -> None:
        """ Calculate medians and robust median absolute deviations for each feature (column) of the data,
        and store them within the object's attributes.

        Parameters
        ----------
            data : numpy.ndarray
                Input array of size *[n,m]* where *n* is the number of datapoints and *m* the number of features
        """
        # check whether array is 2-D
        if len(data.shape) != 2:
            raise IndexError('Provide 2-D array. First dimension is datapoints and second dimension is features.')

        # compute the medians of array column values
        self.median = np.median(a=data, axis=0)

        # in each array column compute the median absolute deviation from the column median value
        # mad = median(abs(a - median_value)/norm_constant); where norm constant is scipy.stats.norm.ppf(3/4.) = 0.6745
        # Robust methods reduce the influence of the outlier values and the long tails in distributions,
        # thus providing statistics that describe the distribution of the central or “good” part of the data collected.
        # Further details: <https://consultglp.com/wp-content/uploads/2015/02/robust-statistics-mad-method.pdf>
        self.mad = mad(a=data, axis=0)

        # set object as fitted
        self.fitted = True
        
    def fit_transform(self, data) -> numpy.ndarray:
        """ Calculate medians and robust median absolute deviations for each feature (column) of the data.
        And then calculate normalized values for each feature (column) of the data
        using median and robust median absolute deviation values.

        Parameters
        ----------
            data : numpy.ndarray
                Input array of size *[n,m]* where *n* is the number of datapoints and *m* the number of features
        """
        # medians and robust median absolute deviations for each feature
        self.fit(data)

        # calculate normalized values
        return self.transform(data)
    
    def transform(self, data) -> numpy.ndarray:
        """ Calculate normalized values for each feature (column) of the data
        using median and robust median absolute deviation values.

        Parameters
        ----------
        data : numpy.ndarray
            Input array of size *[n,m]* where *n* is the number of datapoints and *m* the number of features
        """
        if not self.fitted:
            raise TypeError('The scaler has not been yet fitted. Call fit or fit_transform!')

        if len(data.shape) != 2:
            raise IndexError('Provide 2-D array. First dimension is datapoints and second dimension is features.')

        # create a new array of given shape, filled with zeros
        transformed_data = np.zeros(shape=data.shape)

        # iterate over columns of the input array
        # and normalize values using median and robust median absolute deviation
        for i in range(data.shape[1]):
            transformed_data[:, i] = (data[:, i] - self.median[i]) / self.mad[i]

        return transformed_data

    def inverse_transform(self, data) -> numpy.ndarray:
        """ Calculate denormalized values for each feature (column) of the data
        using median and robust median absolute deviation values.

        Parameters
        ----------
        data : numpy.ndarray
            Input array of size *[n,m]* where *n* is the number of datapoints and *m* the number of features
        """
        if not self.fitted:
            raise TypeError('The scaler has not been yet fitted. Call fit or fit_transform!')

        if len(data.shape) != 2:
            raise IndexError('Provide 2-D array. First dimension is datapoints and second dimension is features.')

        # create a new array of given shape, filled with zeros
        transformed_data = np.zeros(shape=data.shape)

        # iterate over columns of the input array
        # and denormalize values using median and robust median absolute deviation
        for i in range(data.shape[1]):
            transformed_data[:, i] = data[:, i] * self.mad[i] + self.median[i] 

        return transformed_data


class InvariantScaler(MedianScaler):
    """ A class to normalize and denormalize data using median and robust median absolute deviation values
    plus inverse hyperbolic sine (sinh^-1.) transformation. """
    def __init__(self):
        super().__init__()

    def fit_transform(self, data):
        super().fit(data)
        return self.transform(data)
    
    def transform(self, data) -> numpy.ndarray:
        """ Calculate the inverse hyperbolic sine (sinh^-1.) of the normalized values
        for each feature (column) of the data using median and robust median absolute deviation values.

        Parameters
        ----------
            data : numpy.ndarray
                Input array of size *[n,m]* where *n* is the number of datapoints and *m* the number of features
        """
        transformed_data = np.arcsinh(super().transform(data))
        return transformed_data

    def inverse_transform(self, data):
        """ Calculate denormalized values for each feature (column) of the data
        using median and robust median absolute deviation values and sinh() function.

        Parameters
        ----------
            data : numpy.ndarray
                Input array of size *[n,m]* where *n* is the number of datapoints and *m* the number of features
        """
        transformed_data = super().inverse_transform(np.sinh(data))
        return transformed_data


class DataScaler(object):
    """Class to perform data scaling operations, which follows the same syntax of the scalers defined in the
    `sklearn.preprocessing <https://scikit-learn.org/stable/modules/preprocessing.html>`_ module of the
    scikit-learn library.

    The scaling technique is defined by the ``normalize`` parameter which takes one of the following values:

    - ``'Norm'`` for normalizing the data to the interval [0, 1].
    - ``'Norm1'`` for normalizing the data to the interval [-1, 1].
    - ``'Std'`` for standarizing the data to follow a normal distribution.
    - ``'Median'`` for normalizing the data based on the median
      as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.
    - ``'Invariant'`` for scaling the data based on the asinh (sinh^-1) variance stabilizing transformations
      as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.

    Parameters
    ----------
        normalize : str
            Type of scaling to be performed.
            Possible values are

                - ``'Norm'``
                - ``'Norm1'``
                - ``'Std'``
                - ``'Median'``
                - ``'Invariant'``

    Example
    --------
        >>> from epftoolbox.data import read_and_split_data
        >>> from epftoolbox.data import DataScaler
        >>> df_train, df_test = read_and_split_data(path='.', dataset='PJM', response='Zonal COMED price',
        ...                                         begin_test_date='01-01-2016', end_test_date='01-02-2016')
        Test datasets: 2016-01-01 00:00:00 - 2016-02-01 23:00:00
        >>> df_train.tail()
                                 Price  Exogenous 1  Exogenous 2
        Date
        2015-12-31 19:00:00  29.513832     100700.0      13015.0
        2015-12-31 20:00:00  28.440134      99832.0      12858.0
        2015-12-31 21:00:00  26.701700      97033.0      12626.0
        2015-12-31 22:00:00  23.262253      92022.0      12176.0
        2015-12-31 23:00:00  22.262431      86295.0      11434.0
        >>> df_test.head()
                                 Price  Exogenous 1  Exogenous 2
        Date
        2016-01-01 00:00:00  20.341321      76840.0      10406.0
        2016-01-01 01:00:00  19.462741      74819.0      10075.0
        2016-01-01 02:00:00  17.172706      73182.0       9795.0
        2016-01-01 03:00:00  16.963876      72300.0       9632.0
        2016-01-01 04:00:00  17.403722      72535.0       9566.0
        >>> x_train = df_train.values
        >>> x_test = df_train.values
        >>> scaler = DataScaler('Norm')
        >>> x_train_scaled = scaler.fit_transform(x_train)
        >>> x_test_scaled = scaler.transform(x_test)
        >>> x_train_inverse = scaler.inverse_transform(x_train_scaled)
        >>> x_test_inverse = scaler.inverse_transform(x_test_scaled)
        >>> x_train[:3,:]
        array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
               [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
               [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
        >>> x_train_scaled[:3,:]
        array([[0.03833877, 0.2736787 , 0.28415155],
               [0.03608228, 0.24425597, 0.24633138],
               [0.03438982, 0.23016409, 0.2261206 ]])
        >>> x_train_inverse[:3,:]
        array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
               [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
               [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
        >>> x_test[:3,:]
        array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
               [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
               [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
        >>> x_test_scaled[:3,:]
        array([[0.03833877, 0.2736787 , 0.28415155],
               [0.03608228, 0.24425597, 0.24633138],
               [0.03438982, 0.23016409, 0.2261206 ]])
        >>> x_test_inverse[:3,:]
        array([[2.5464211e+01, 8.5049000e+04, 1.1509000e+04],
               [2.3554578e+01, 8.2128000e+04, 1.0942000e+04],
               [2.2122277e+01, 8.0729000e+04, 1.0639000e+04]])
    """
    def __init__(self, normalize):
        if normalize == 'Norm':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif normalize == 'Norm1':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif normalize == 'Std':
            self.scaler = StandardScaler()
        elif normalize == 'Median':
            self.scaler = MedianScaler()
        elif normalize == 'Invariant':
            self.scaler = InvariantScaler()        

    def fit(self, dataset):
        """ Method that estimates the scaler based on the ``dataset``.

        Parameters
        ----------
            dataset : numpy.ndarray
                Dataset used to estimate the scaler
        """
        self.scaler.fit(dataset)

    def fit_transform(self, dataset):
        """ Method that - according to scaler setting - calculates normalized values
        for each feature (column) of the ``dataset``.
        
        Parameters
        ----------
            dataset : numpy.ndarray
                Dataset used to estimate the scaler
        
        Returns
        -------
            numpy.ndarray
                Scaled (normalized) data
        """
        return self.scaler.fit_transform(dataset)

    def transform(self, dataset):
        """Method that scales the data in ``dataset``.
        
        To estimate the scaler, the :class:`fit_transform` method must be called
        before calling the :class:`transform` method.
        Parameters
        ----------
            dataset : numpy.ndarray
                Dataset to be scaled
        
        Returns
        -------
            numpy.ndarray
                Scaled data
        """
        return self.scaler.transform(dataset)

    def inverse_transform(self, dataset):
        """Method that inverse-scale the data in ``dataset``
        
        To estimate the scaler, the :class:`fit_transform` method must be called
        before calling the :class:`inverse_transform` method.

        Parameters
        ----------
            dataset : numpy.ndarray
                Dataset to be scaled
        
        Returns
        -------
            numpy.ndarray
                Inverse-scaled data
        """
        return self.scaler.inverse_transform(dataset)


def scaling(datasets, normalize):
    """ Scales data and returns the scaled data and the :class:`DataScaler` used for scaling.

    It rescales all the datasets contained in the list ``datasets`` using the first dataset as reference. 
    For example, if ``datasets=[X_1, X_2, X_3]``, the function estimates a :class:`DataScaler` object
    using the array ``X_1``, and transform ``X_1``, ``X_2``, and ``X_3`` using the :class:`DataScaler` object.

    Each dataset must be a numpy.ndarray, and it should have the same column-dimensions.
    For example, if ``datasets=[X_1, X_2, X_3]``, ``X_1`` must be a numpy.ndarray of size ``[n_1, m]``,
    ``X_2`` of size ``[n_2, m]``, and ``X_3`` of size ``[n_3, m]``, where ``n_1``, ``n_2``, ``n_3`` can be
    different.

    The scaling technique is defined by the ``normalize`` parameter which takes one of the 
    following values: 

    - ``'Norm'`` for normalizing the data to the interval [0, 1].
    - ``'Norm1'`` for normalizing the data to the interval [-1, 1].
    - ``'Std'`` for standarizing the data to follow a normal distribution.
    - ``'Median'`` for normalizing the data based on the median
      as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.
    - ``'Invariant'`` for scaling the data based on the asinh (sinh^-1) variance stabilizing transformation
      as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.

    The function returns the scaled data together with a :class:`DataScaler` object representing the scaling. 
    This object can be used to scale another dataset using the same rules or to inverse-transform the data.
    
    Parameters
    ----------
        datasets : list
            List of numpy.ndarray objects to be scaled.
        normalize : str
            Type of scaling to be performed.
            Possible values are
                - ``'Norm'``
                - ``'Norm1'``
                - ``'Std'``
                - ``'Median'``
                - ``'Invariant'``
    
    Returns
    -------
        List, :class:`DataScaler`
            List of scaled datasets and the :class:`DataScaler` object used for scaling.
            Each dataset in the list is a numpy.ndarray.
    
    Example
    --------
        >>> from epftoolbox.data import read_and_split_data
        >>> from epftoolbox.data import scaling
        >>> df_train, df_test = read_and_split_data(path='../examples/datasets', dataset='PJM',
        ...                                         response='Zonal COMED price',
        ...                                         begin_test_date='01-01-2016 00:00',
        ...                                         end_test_date='01-02-2016 23:00')
        Training dataset period: 2013-01-01 00:00:00 - 2015-12-31 23:00:00
        Testing dataset period: 2016-01-01 00:00:00 - 2016-02-01 23:00:00
        Test datasets: 2016-01-01 00:00:00 - 2016-02-01 23:00:00
        >>> print('\\ndf_train.tail()', df_train.tail(), sep='\\n')
        df_train.tail()
                                 Price  Exogenous 1  Exogenous 2
        Date
        2015-12-31 19:00:00  29.513832     100700.0      13015.0
        2015-12-31 20:00:00  28.440134      99832.0      12858.0
        2015-12-31 21:00:00  26.701700      97033.0      12626.0
        2015-12-31 22:00:00  23.262253      92022.0      12176.0
        2015-12-31 23:00:00  22.262431      86295.0      11434.0
        >>> print('\\ndf_test.head()', df_test.head(), sep='\\n')
        df_test.head()
                                 Price  Exogenous 1  Exogenous 2
        Date
        2016-01-01 00:00:00  20.341321      76840.0      10406.0
        2016-01-01 01:00:00  19.462741      74819.0      10075.0
        2016-01-01 02:00:00  17.172706      73182.0       9795.0
        2016-01-01 03:00:00  16.963876      72300.0       9632.0
        2016-01-01 04:00:00  17.403722      72535.0       9566.0
        >>> x_train = df_train.values
        >>> x_test = df_train.values
        >>> [x_train_scaled_Norm, x_test_scaled_Norm], scaler = scaling(datasets=[x_train, x_test], normalize='Norm')
        >>> print('\\nx_train[:4, :]:', x_train[:4, :], sep='\\n')
        x_train[:4, :]:
        [[2.5464211e+01 8.5049000e+04 1.1509000e+04]
         [2.3554578e+01 8.2128000e+04 1.0942000e+04]
         [2.2122277e+01 8.0729000e+04 1.0639000e+04]
         [2.1592066e+01 8.0248000e+04 1.0476000e+04]]
        >>> print('\\nx_train_scaled_Norm[:4, :]:', x_train_scaled_Norm[:4, :], sep='\\n')
        x_train_scaled_Norm[:4, :]:
        [[0.03833877 0.2736787  0.28415155]
         [0.03608228 0.24425597 0.24633138]
         [0.03438982 0.23016409 0.2261206 ]
         [0.0337633  0.22531906 0.21524813]]
        >>> print('\\nx_test[:4, :]:', x_test[:4, :], sep='\\n')
        x_test[:4, :]:
        [[2.5464211e+01 8.5049000e+04 1.1509000e+04]
         [2.3554578e+01 8.2128000e+04 1.0942000e+04]
         [2.2122277e+01 8.0729000e+04 1.0639000e+04]
         [2.1592066e+01 8.0248000e+04 1.0476000e+04]]
        >>> print('\\nx_test_scaled_Norm[:4, :]:', x_test_scaled_Norm[:4, :], sep='\\n')
        x_test_scaled_Norm[:4, :]:
        [[0.03833877 0.2736787  0.28415155]
         [0.03608228 0.24425597 0.24633138]
         [0.03438982 0.23016409 0.2261206 ]
         [0.0337633  0.22531906 0.21524813]]
        >>> print('\\ntype(scaler):', type(scaler))
        type(scaler): <class 'epftoolbox.data._wrangling.DataScaler'>
    """
    # instantiate a scaler object according to the normalization technique
    scaler = DataScaler(normalize=normalize)

    # estimate the scaler based on the first dataset
    scaler.fit(dataset=datasets[0])

    # scale (normalize) all the datasets in the list
    scaled_datasets = [scaler.transform(dataset=dataset) for dataset in datasets]

    return scaled_datasets, scaler


if __name__ == '__main__':
    from epftoolbox.data import read_and_split_data
    from epftoolbox.data import scaling
    df_train, df_test = read_and_split_data(path='../examples/datasets', dataset='PJM', response='Zonal COMED price',
                                            begin_test_date='01-01-2016 00:00', end_test_date='01-02-2016 23:00')
    print('\ndf_train.tail()', df_train.tail(), sep='\n')
    print('\ndf_test.head()', df_test.head(), sep='\n')

    x_train = df_train.values
    x_test = df_train.values

    [x_train_scaled_Norm, x_test_scaled_Norm], scaler = scaling(datasets=[x_train, x_test], normalize='Norm')

    print('\nx_train[:4, :]:', x_train[:4, :], sep='\n')
    print('\nx_train_scaled_Norm[:4, :]:', x_train_scaled_Norm[:4, :], sep='\n')
    print('\nx_test[:4, :]:', x_test[:4, :], sep='\n')
    print('\nx_test_scaled_Norm[:4, :]:', x_test_scaled_Norm[:4, :], sep='\n')
    print('\ntype(scaler):', type(scaler))
    print('\nscaler.scaler:', scaler.scaler, sep='\n')
