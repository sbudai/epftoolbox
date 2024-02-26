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
                        years_test=2, begin_test_date=None, end_test_date=None):
    """ Import the day-ahead electricity price data_ supplemented with exogenous explanatory variables,
    and provides a split between training and testing dataset based on the related arguments provided.
    It also renames the columns of the training and testing dataset to match the requirements of the
    current module's prediction models.
    Namely, assuming that there are N exogenous explanatory variable inputs,
    the columns of the resulting dataframes will be named as
    ['Price', 'Exogenous_1', 'Exogenous_2', ..., 'Exogenous_N'].

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
                PJM - 'Zonal COMED price'
                NP - 'Price'
                BE - 'Prices'
                FR - 'Prices'
                DE - 'Price'
            years_test : int
                Optional parameter to set the number of years of the being test dataset,
                counting back from the end of the input dataset.
                Note that a year is considered to be 364 days.
                It is only used if the arguments `begin_test_date` and `end_test_date` are not provided.
            begin_test_date : datetime.datetime | str | None
                Optional parameter to select the test dataset.
                Used in combination with the argument `end_test_date`.
                If either of them is not provided, the test dataset will be split using the `years_test` argument.
                The `begin_test_date` should either be a string with the following format "%d/%m/%Y 00:00",
                or a datetime object.
            end_test_date : datetime.datetime | str | None
                Optional parameter to select the test dataset.
                This value may be earlier than the last timestamp in the input dataset.
                Used in combination with the argument `begin_test_date`.
                If either of them is not provided, the test dataset will be split using the `years_test` argument.
                The `end_test_date` should either be a string with the following format "%d/%m/%Y 23:00",
                or a datetime object.

        Returns
        -------
            tuple[pandas.DataFrame, pandas.DataFrame]
                Training dataset, testing dataset
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
    df_.rename(columns={all_columns[i]: 'Exogenous_{0}'.format(j + 1) for j, i in enumerate(exogen_indices)},
               inplace=True)
    df_.columns.name = None

    # set the index name
    df_.index.name = 'date'

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
    # These codes below will only be executed if this script is run as the main program.
    df_train, df_test = read_and_split_data(
        path=os.path.join('..', '..', 'examples', 'datasets'),
        dataset='DE',
        response='Price',
        years_test=2,
        # begin_test_date="01/01/2016 00:00",
        # end_test_date="01/02/2016 23:00",
    )
    # Training dataset period: 2012-01-09 00:00:00 - 2016-01-03 23:00:00
    # Testing dataset period: 2016-01-04 00:00:00 - 2017-12-31 23:00:00

    print("df_train:", df_train, sep="\n")
    # df_train:
    #                      Price  Exogenous_1  Exogenous_2
    # date
    # 2012-01-09 00:00:00  34.97     16382.00   3569.52750
    # 2012-01-09 01:00:00  33.43     15410.50   3315.27500
    # 2012-01-09 02:00:00  32.74     15595.00   3107.30750
    # 2012-01-09 03:00:00  32.46     16521.00   2944.62000
    # 2012-01-09 04:00:00  32.50     17700.75   2897.15000
    # ...                    ...          ...          ...
    # 2016-01-03 19:00:00  27.08     19242.25  18583.11375
    # 2016-01-03 20:00:00  25.33     18620.00  18589.18450
    # 2016-01-03 21:00:00  22.11     18490.25  18550.75675
    # 2016-01-03 22:00:00  20.91     18717.50  18612.22950
    # 2016-01-03 23:00:00  14.43     17559.25  18580.60000
    #
    # [34944 rows x 3 columns]

    print("df_test:", df_test, sep="\n")
    # df_test:
    #                      Price  Exogenous_1  Exogenous_2
    # date
    # 2016-01-04 00:00:00  13.78     16077.75  20162.33750
    # 2016-01-04 01:00:00  12.77     15573.25  19991.67550
    # 2016-01-04 02:00:00  10.56     15373.50  19701.83700
    # 2016-01-04 03:00:00   3.87     15278.50  19222.00550
    # 2016-01-04 04:00:00   3.20     15505.50  18784.78175
    # ...                    ...          ...          ...
    # 2017-12-31 19:00:00   7.92     16601.00  30649.08950
    # 2017-12-31 20:00:00   4.06     15977.75  30034.54300
    # 2017-12-31 21:00:00   5.30     15715.00  29653.00775
    # 2017-12-31 22:00:00   1.86     15876.00  29520.32950
    # 2017-12-31 23:00:00  -0.92     15130.00  29466.40875
    #
    # [17472 rows x 3 columns]
