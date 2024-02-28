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
                        sep=',', encoding='utf-8', index_col=0, years_test=2, begin_test_date=None, end_test_date=None,
                        summary=False):
    """ Import the day-ahead electricity price data supplemented with exogenous explanatory variables,
    and provides a split between training and testing dataset based on the related arguments provided.
    It also renames the columns of the training and testing dataset to match the requirements of the
    current module's prediction models.
    Namely, assuming that there are N exogenous explanatory variable inputs,
    the columns of the resulting dataframes will be named as
    ['Price', 'Exogenous_1', 'Exogenous_2', ..., 'Exogenous_N'].

    The indices are naive timestamps in the format "%Y-%m-%d %H:%M:%S", and denote the starting
    timestamp of the related electricity delivery period.
    Those timestamps basically should be interpreted as the local time of the corresponding market.
    On those markets, which use Daylight Saving Time (DST), an apriori data preparation is required.
    When DST observation begins, the clocks are advanced by one hour during the very early morning.
    When DST observation ends and the standard time observation resumes,
    the clocks are turned back one hour during the very early morning.
    Hence, there is a missing hour in the dataset in every spring, and there is a duplicate hour in every autumn.
    The models in this module cannot handle well such missing and duplicate hours,
    so the missing hour should be imputed and the duplicate hour should be averaged in advance.

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
        sep : str
            Delimiter of the input dataset.
        encoding : str
            Encoding of the input dataset.
        index_col : int | str
            A Column of the input dataset to use as row label,
            denoted either by column label or column index.
            The denoted column should contain a datetime-like object
            or a collection of such objects that can be converted to datetime.
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
        summary : bool
            Whether to print a summary of the resulting DataFrames.

    Returns
    -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            Training dataset, testing dataset
    """

    # Check if index_col can be valid
    if index_col is None:
        raise ValueError('There must be an index column in the input dataset.')
    elif isinstance(index_col, (int, str)):
        pass
    else:
        raise ValueError('`index_col` should be either a single integer or a single string.')

    # Check if the provided directory exists on local, and if not, create it
    os.makedirs(name=path, exist_ok=True)

    # Compose the local path of the input dataset
    file_path = os.path.join(path, '{0}.csv'.format(dataset))

    # If the nominated dataset is one of the standard open-access benchmark ones
    # and not yet downloaded into the local input folder, then it will be downloaded.
    if dataset in ['PJM', 'NP', 'FR', 'BE', 'DE'] and not os.path.exists(file_path):
        file_url = 'https://zenodo.org/records/4624805/files/{0}.csv'.format(dataset)
        req = requests.get(url=file_url)
        with open(file=file_path, mode='wb') as f:
            f.write(req.content)

    # Import the input dataset from local disk
    try:
        df_ = pd.read_csv(filepath_or_buffer=file_path, sep=sep, encoding=encoding, index_col=index_col)
    except FileNotFoundError as error:
        raise error

    # check if the input dataset is empty
    if df_.empty:
        raise ValueError('The input dataset should not be empty.')

    # check if all columns are numeric
    if df_.shape != df_.select_dtypes(include=np.number).shape:
        print(df_.info())
        raise ValueError('The input dataset should only contain numeric values.')

    # converts indices to a pandas datetime
    df_.index = pd.to_datetime(arg=df_.index)

    # sort by the index if necessary
    if not df_.index.is_monotonic_increasing:
        df_.sort_index(inplace=True)

    # check if the input dataset has no duplicate indices
    if df_.index.has_duplicates:
        raise IndexError('The input dataset should not have duplicate indices.')

    # check if the input dataset has a consistent frequency (no missing records)
    if not df_.index.inferred_freq:
        raise IndexError('The indices of the input dataset should have a consistent frequency.')

    # sort the dataset by the index if it is not sorted yet
    if not df_.index.is_monotonic_increasing:
        df_ = df_.sort_index()

    # extract all column names
    orig_col_names = df_.columns.values.tolist()

    # detect the index of the response variable
    response_index = [key for key, value in enumerate(orig_col_names) if value.strip() == response]
    if len(response_index) == 0:
        raise ValueError('The response variable does not exist in the dataset.')

    # extract the column indices of the exogenous variables
    exogen_indices = [i for i in range(len(orig_col_names)) if i not in response_index]

    # rename all the columns accordingly
    df_.rename(columns={orig_col_names[i]: 'Price' for i in response_index},
               inplace=True)
    df_.rename(columns={orig_col_names[i]: 'Exogenous_{0}'.format(j + 1) for j, i in enumerate(exogen_indices)},
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

    else:
        raise Exception("Please provide either the number of years for testing "
                        "or the start and end dates of the test period!")

    # check whether begin and end dates are in the right order
    if begin_test_date > end_test_date:
        raise ValueError("The test period opening should be before its ending.\n"
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
        raise ValueError("The test period opening should be before the end of the input dataset.")

    if begin_test_date <= df_.index[0]:
        raise ValueError("The test period opening should be after the beginning of the input dataset.")

    # Split the dataset into training and test datasets
    df_train_ = df_.loc[:begin_test_date - pd.Timedelta(hours=1), :]
    df_test_ = df_.loc[begin_test_date:end_test_date, :]
    print('Training dataset period: {0} - {1}'.format(df_train_.index[0], df_train_.index[-1]))
    print('Testing dataset period: {0} - {1}'.format(df_test_.index[0], df_test_.index[-1]))

    if summary:
        # print the column renaming rules
        print()
        print("header renamed as:")
        new_col_names = df_.columns.values.tolist()
        for ind, orig_col_name in enumerate(orig_col_names):
            print("'{0}' -> '{1}'".format(orig_col_name, new_col_names[ind]))

        # show summary statistics of df_train_ and df_test_
        for df_name_, df_t_ in {"df_train": df_train_, "df_test": df_test_}.items():
            print()
            print("{0} summary:".format(df_name_), df_t_.index._summary(),
                  "inferred frequency of the index: {0}".format(df_t_.index.inferred_freq),
                  df_t_.describe(include="all"), sep="\n")

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
        summary=True,
    )
    # Training dataset period: 2012-01-09 00:00:00 - 2016-01-03 23:00:00
    # Testing dataset period: 2016-01-04 00:00:00 - 2017-12-31 23:00:00
    #
    # header renamed as:
    # 'Price' -> 'Price'
    # 'Ampirion Load Forecast' -> 'Exogenous_1'
    # 'PV+Wind Forecast' -> 'Exogenous_2'
    #
    # df_train summary:
    # DatetimeIndex: 34944 entries, 2012-01-09 00:00:00 to 2016-01-03 23:00:00
    # inferred frequency of the index: h
    #               Price   Exogenous_1   Exogenous_2
    # count  34944.000000  34944.000000  34944.000000
    # mean      36.197639  21499.904762   9388.916649
    # std       15.946342   3844.859619   6734.883630
    # min     -221.990000  11508.000000    302.237500
    # 25%       27.507500  18343.687500   3936.407625
    # 50%       34.970000  21432.500000   7708.808125
    # 75%       45.070000  24646.812500  13524.615438
    # max      210.000000  35499.250000  39344.662250
    #
    # df_test summary:
    # DatetimeIndex: 17472 entries, 2016-01-04 00:00:00 to 2017-12-31 23:00:00
    # inferred frequency of the index: h
    #               Price   Exogenous_1   Exogenous_2
    # count  17472.000000  17472.000000  17472.000000
    # mean      31.638297  21065.931834  13350.962922
    # std       15.486453   3515.697442   8608.720628
    # min     -130.090000  10718.250000    490.969750
    # 25%       24.040000  18143.687500   6331.621625
    # 50%       31.100000  21030.375000  11649.831375
    # 75%       38.150000  24125.562500  18841.585250
    # max      163.520000  28813.750000  48587.881000
