"""
Function to read electricity market data either locally or from an online database
"""

# Author: Jesus Lago & Sandor Budai

# License: AGPL-3.0 License


import pandas as pd
import numpy as np
import math
import os

# import pandas.errors
import requests


def read_and_split_data(path=os.path.join('.', 'datasets'), dataset='PJM', response='Zonal COMED price',
                        years_test=2, begin_test_date=None, end_test_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Import the day-ahead electricity price data_ supplemented with exogenous explanatory variables,
    and provides a split between training and testing dataset based on the related arguments provided.
    It also renames the columns of the training and testing dataset to match the requirements of the
    current module's prediction models.
    Namely, assuming that there are N exogenous explanatory variable inputs,
    the columns of the resulting dataframes will be naTest datasetsmed as
    ['Price', 'Exogenous 1', 'Exogenous 2', ..., 'Exogenous N'].

    If `dataset` is one of the online available standard market datasets of the study,
    such as "PJM", "NP", "BE", "FR" or "DE", the function checks
    whether the `XY.csv` already exists in the `path` folder.
    If not, it downloads the data_ from <https://zenodo.org/records/4624805> and saves it into there.
        - "PJM" refers to the Pennsylvania-New Jersey-Maryland day-ahead market,
        - "NP" refers to the Nord Pool day-ahead market,
        - "BE" refers to the EPEX-Belgium day-ahead market
        - "FR" refers to the EPEX-France day-ahead market
        - "DE" refers to the EPEX-Germany day-ahead market
    Note, that the data_ available online for these five markets is limited to certain periods.
    
        Parameters
        ----------
            path : str
                Path of the local folder where the input dataset should be placed locally.
            dataset : str
                Name of the file (w/o "csv" extension) containing the input dataset.
            response : str
                Name of the column in the input dataset that denotes the response variable.
            years_test : int
                Optional parameter to set the number of years of the being test dataset,
                counting back from the end of the input dataset.
                Note that a year is considered to be 364 days.
                It is only used if the arguments `begin_test_date` and `end_test_date` are not provided.
            begin_test_date : datetime | str
                Optional parameter to select the test dataset.
                Used in combination with the argument `end_test_date`.
                If either of them is not provided, the test dataset will be split using the `years_test` argument.
                The `begin_test_date` should either be a string with the following format "%d/%m/%Y 00:00",
                or a datetime object.
            end_test_date : datetime | str
                Optional parameter to select the test dataset.
                This value may be earlier than the last timestamp in the input dataset.
                Used in combination with the argument `begin_test_date`.
                If either of them is not provided, the test dataset will be split using the `years_test` argument.
                The `end_test_date` should either be a string with the following format "%d/%m/%Y 23:00",
                or a datetime object.
        Returns
        -------
            pandas.DataFrame, pandas.DataFrame
                Training dataset, testing dataset

        Example
        --------
            >>> from epftoolbox.data import read_and_split_data
            >>> df_train, df_test = read_and_split_data(
            ...     path=os.path.join('.', 'datasets'),
            ...     dataset='PJM',
            ...     response='Zonal COMED price',
            ...     begin_test_date='01/01/2016 00:00',
            ...     end_test_date='01/02/2016 23:00',
            ... )
            Training dataset period: 2013-01-01 00:00:00 - 2015-12-31 23:00:00
            Testing dataset period: 2016-01-01 00:00:00 - 2016-02-01 23:00:00
            >>> print("df_train:", df_train, sep="\\n")
            df_train:
                                     Price  Exogenous 1  Exogenous 2
            Date
            2013-01-01 00:00:00  25.464211      85049.0      11509.0
            2013-01-01 01:00:00  23.554578      82128.0      10942.0
            2013-01-01 02:00:00  22.122277      80729.0      10639.0
            2013-01-01 03:00:00  21.592066      80248.0      10476.0
            2013-01-01 04:00:00  21.546501      80850.0      10445.0
            ...                        ...          ...          ...
            2015-12-31 19:00:00  29.513832     100700.0      13015.0
            2015-12-31 20:00:00  28.440134      99832.0      12858.0
            2015-12-31 21:00:00  26.701700      97033.0      12626.0
            2015-12-31 22:00:00  23.262253      92022.0      12176.0
            2015-12-31 23:00:00  22.262431      86295.0      11434.0

            [26280 rows x 3 columns]

            >>> print("df_test:", df_test, sep="\\n")
            df_test:
                                     Price  Exogenous 1  Exogenous 2
            Date
            2016-01-01 00:00:00  20.341321      76840.0      10406.0
            2016-01-01 01:00:00  19.462741      74819.0      10075.0
            2016-01-01 02:00:00  17.172706      73182.0       9795.0
            2016-01-01 03:00:00  16.963876      72300.0       9632.0
            2016-01-01 04:00:00  17.403722      72535.0       9566.0
            ...                        ...          ...          ...
            2016-02-01 19:00:00  28.056729      99400.0      12680.0
            2016-02-01 20:00:00  26.916456      97553.0      12495.0
            2016-02-01 21:00:00  24.041505      93983.0      12267.0
            2016-02-01 22:00:00  22.044896      88535.0      11747.0
            2016-02-01 23:00:00  20.593339      82900.0      10974.0

            [768 rows x 3 columns]

    """

    # Checking if provided directory exists on local, and if not, create it
    os.makedirs(name=path, exist_ok=True)

    # Compose the local path of the input dataset
    file_path = os.path.join(path, dataset + '.csv')

    # If the nominated dataset is one of the standard open-access benchmark ones
    # and not yet downloaded into the local input folder, then it will be downloaded.
    if dataset in ['PJM', 'NP', 'FR', 'BE', 'DE'] and not os.path.exists(file_path):
        file_url = 'https://zenodo.org/records/4624805/files/{0}.csv'.format(dataset)
        req = requests.get(url=file_url)
        with open(file=file_path, mode='wb') as f:
            f.write(req.content)

    # Import the input dataset from local disk
    try:
        df_ = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError as error:
        raise error

    # converts indices to a pandas datetime
    df_.index = pd.to_datetime(arg=df_.index)

    # extract all column names
    all_columns = df_.columns.values.tolist()

    # detect the index of the response variable
    response_index = [key for key, value in enumerate(all_columns) if value.strip() == response]
    if len(response_index) == 0:
        raise ValueError('The response variable does not exist in the dataset.')

    # extract the indices of the exogenous variables
    exogen_indices = [i for i in range(len(all_columns)) if i not in response_index]

    # rename all the columns accordingly
    df_.rename(columns={all_columns[i]: 'Price' for i in response_index},
               inplace=True)
    df_.rename(columns={all_columns[i]: 'Exogenous {0}'.format(j + 1) for j, i in enumerate(exogen_indices)},
               inplace=True)

    # The training and test datasets can be defined by providing a number of years for testing
    # or by providing the start and end date of the test period
    if begin_test_date is None and end_test_date is None and years_test is not None:

        if not isinstance(years_test, (float, int)):
            raise TypeError("The years_test should be numerical object")

        if years_test <= 0:
            raise ValueError("The years_test should be positive!")

        # We consider that a year is 52 weeks (364 days) instead of the traditional 365
        last_23h_index = np.where(df_.index.hour == 23)[0][-1]
        end_test_date = df_.index[last_23h_index]
        begin_test_date = end_test_date - pd.Timedelta(days=math.ceil(364 * years_test)) + pd.Timedelta(hours=1)

    elif begin_test_date is not None and end_test_date is not None:

        if not isinstance(begin_test_date, (pd.Timestamp, str)):
            raise TypeError("The begin_test_date should be a pandas Timestamp object"
                            " or a string with '%d/%m/%Y 00:00' format.")

        if not isinstance(end_test_date, (pd.Timestamp, str)):
            raise TypeError("The end_test_date should be a pandas Timestamp object"
                            " or a string with '%d/%m/%Y 23:00' format.")

        # Convert dates to pandas Timestamp
        begin_test_date = pd.to_datetime(begin_test_date, dayfirst=True)
        end_test_date = pd.to_datetime(end_test_date, dayfirst=True)

        # check whether begin and end dates are in the right order
        if begin_test_date > end_test_date:
            raise ValueError("The begin_test_date should be before the end_test_date.\n"
                             "{0} is not earlier than {1}.".format(begin_test_date, end_test_date))

        # Check if begin_test_date is at midnight
        if not begin_test_date.hour == 0:
            raise ValueError("The test period opening hour should be 00:00 instead of {0}:{1}.".
                             format(begin_test_date.hour, begin_test_date.minute))

        # Check if end_test_date is at one hour before midnight
        if not end_test_date.hour == 23:
            raise ValueError("The test period closing hour should be 23:00 instead of {0}:{1}.".
                             format(end_test_date.hour, end_test_date.minute))

        if begin_test_date >= df_.index[-1]:
            raise ValueError("The begin_test_date should be before the end of the dataset.")

        if begin_test_date <= df_.index[0]:
            raise ValueError("The begin_test_date should be after the beginning of the dataset.")

    else:
        raise Exception("Please provide either the number of years for testing "
                        "or the start and end dates of the test period!")

    # Split the dataset into training and test datasets
    df_train_ = df_.loc[:begin_test_date - pd.Timedelta(hours=1), :]
    df_test_ = df_.loc[begin_test_date:end_test_date, :]
    print('Training dataset period: {0} - {1}'.format(df_train_.index[0], df_train_.index[-1]))
    print('Testing dataset period: {0} - {1}'.format(df_test_.index[0], df_test_.index[-1]))

    return df_train_, df_test_


if __name__ == '__main__':
    df_train, df_test = read_and_split_data(
        path=os.path.join('..', '..', 'datasets'),
        dataset='PJM',
        response='Zonal COMED price',
        # years_test=2,
        begin_test_date="01/01/2016 00:00",
        end_test_date="01/02/2016 23:00",
    )
    print("df_train:", df_train, sep="\n")
    print("df_test:", df_test, sep="\n")
