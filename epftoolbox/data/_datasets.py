"""
Function to read electricity market data either locally or from an online database
"""

# Author: Jesus Lago & Sandor Budai

# License: AGPL-3.0 License


import pandas as pd
import numpy as np
import math
import os
import csv
import requests


def _arrange_index(dataset, index_tz, market_tz, intended_freq='1h') -> pd.DataFrame:
    """ Utility function around datetime index issues.
    Such as:

    - If the timezone in which the index values are expressed is different from the timezone of the market, then it
      converts them according to the market timezone.
    - If the index is expressed in timezone-aware format, then it converts the index to timezone-naive format.
    - If the market timezone is in the Daylight Saving Time system, then it checks whether there ar duplicate index
      values, and if so, eliminates duplication by taking the average of the values.
    - If the market timezone is in the Daylight Saving Time system, then it checks whether all the index values
      can be interpreted as valid datetime values.
      If not, it removes the corresponding rows.
    - Check whether the index has monotonous frequency.
      If not, adjust it according to intended frequency using the last valid observation carry forward method.

    Parameters
    ----------
        dataset : pandas.DataFrame
            The dataset which indices and values need to be converted.
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
    """
    # if the index is expressed in timezone-naive format, but the value itself is not UTC, then ...
    if dataset.index[0].tzinfo is None and not index_tz == 'UTC':
        # convert it to timezone-aware format and localize it
        dataset.index = [ind.tz_localize(tz=index_tz, ambiguous=True, nonexistent='NaT') for ind in dataset.index]

        # remove the rows with nonexistent indices
        if len(dataset.index):
            not_valid_index_iloc = np.where(dataset.index.isna())[0]
            not_valid_sample_index_iloc = np.vstack((not_valid_index_iloc - 1,
                                                     not_valid_index_iloc,
                                                     not_valid_index_iloc + 1)).T
            print("\n\t*** These records' with NaT index value are not valid, "
                  "so we remove these rows from the dataset *** ")
            for i in not_valid_sample_index_iloc:
                print(dataset.iloc[i])
                print('-' * 60)

            # remove the rows with not valid index values
            dataset = dataset.loc[dataset.index.notna()]

    # localize the timezone-naive UTC indices as timezone-aware UTC, or convert timezone-aware indices to UTC
    dataset.index = pd.to_datetime(arg=dataset.index, utc=True)

    # convert timezone-aware index from UTC to timezone-aware market_tz time zone
    dataset.index = dataset.index.tz_convert(tz=market_tz)

    # convert the localized indices to timezone-naive localized format
    dataset.index = dataset.index.tz_localize(tz=None)

    # calculate mean values to "duplicate" datetime indices
    # Those records may come from the repeated calendar hour at each autumn in the DST system.
    dataset = dataset.groupby(by=dataset.index).mean()

    # If there is no inferred frequency, then re-index the dataset to comply with the intended frequency.
    # Missing hours most probably come from the missing calendar hour at each spring in the DST system.
    if dataset.index.inferred_freq is None:
        # create a transformed version of the dataset as to conform to a new index with the specified intended_freq
        dataset_w_freq = dataset.asfreq(freq=intended_freq)

        # seek for the freshly created empty rows in the intended frequency conformed dataset
        empty_rows = dataset_w_freq.loc[dataset_w_freq.isnull().any(axis=1)]
        print("\n\t*** The data at these timestamps will be filled with the last valid observation forward method *** ",
              empty_rows, sep="\n")

        # transform the dataset as to conform to a new index with the specified intended_freq
        # and fill the empty rows with the last valid observation carry forward method
        # (note this does not fill NaNs that already were present)
        dataset = dataset.asfreq(freq=intended_freq, method='ffill')

    return dataset


def _detect_col_names(file_path, sep, encoding, index_col):
    """ Detect the column names of the given csv file.
    If there is an index column in the dataset, then its name is excluded from the column name list.

    Parameters
    ----------
        file_path :  str | PathLike
            The path to the csv file.
        sep : str
            The delimiter of the csv file.
        encoding : str
            The encoding of the csv file
        index_col : int | str
            A column in the csv file to use as datetime index,
            denoted either by column label or column index.

    Returns
    -------
        list[str | None]
            The list of detected column names.
    """
    # read only the header of the input dataset
    # hence open the input csv file with the variable name as csv_file
    with open(file=file_path, mode='r', encoding=encoding) as csv_file:
        # extract the first element (row) from csv_reader iterator
        csv_reader = csv.reader(csv_file, delimiter=sep)
        orig_col_names = next((x for x in csv_reader), None)

        # remove the index column from the original column name list
        if isinstance(index_col, str):
            if index_col in orig_col_names:
                orig_col_names.remove(index_col)
            else:
                raise ValueError("The `index_col` '{0}' is not a valid column name.".format(index_col))
        elif isinstance(index_col, int):
            if index_col in range(len(orig_col_names)):
                del orig_col_names[index_col]
            else:
                raise ValueError("The `index_col` '{0}' is not a valid column index.".format(index_col))

    return orig_col_names


def read_data(path=os.path.join('.', 'datasets'), dataset='PJM', index_col=None, response_col=None,
              sep=',', decimal='.', date_format='ISO8601', encoding='utf-8',
              index_tz=None, market_tz=None, intended_freq='1h',
              summary=False):
    """ Import the day-ahead electricity price data supplemented with exogenous explanatory variables.
    It also renames the columns of the dataset to match the requirements of the current module's prediction models.
    Namely, assuming that there are N exogenous explanatory variable inputs,
    the columns of the resulting dataframes will be named as
    ['Price', 'Exogenous_1', 'Exogenous_2', ..., 'Exogenous_N'].

    The returning table's indices are naive timestamps in the format '%Y-%m-%d %H:%M:%S',
    and denote the starting timestamp of the related electricity delivery period.
    Those timestamps basically should be interpreted as the local time of the corresponding market.
    On those markets, which use Daylight Saving Time (DST), an apriori data preparation is required and done.
    When DST observation begins, the clocks are advanced by one hour during the very early morning.
    When DST observation ends and the standard time observation resumes,
    the clocks are turned back one hour during the very early morning.
    Hence, there is a missing hour in the dataset in every spring, and there is a duplicate hour in every autumn.
    The models in this module cannot handle well such missing and duplicate hours,
    so the missing hours should be imputed and the duplicate hours should be averaged in advance.

    If `dataset` is one of the online available standard market datasets of the study,
    such as "PJM", "NP", "BE", "FR" or "DE", the function checks
    whether the `XY.csv` already exists in the `path` folder.
    If not, it downloads the data_ from <https://zenodo.org/records/4624805> and saves it into there.
        - "PJM" refers to the Pennsylvania-New Jersey-Maryland day-ahead market,
        - "NP" refers to the Nord Pool day-ahead market,
        - "BE" refers to the EPEX-Belgium day-ahead market
        - "FR" refers to the EPEX-France day-ahead market
        - "DE" refers to the EPEX-Germany day-ahead market
    Note, that the data available online for these five markets is limited to certain periods.
    
    Parameters
    ----------
        path : str
            Path of the local folder where the input dataset should be placed locally.
        dataset : str
            Name of the csv file (denoted w/o "csv" extension) containing the input dataset.
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
            Transformed dataset
    """

    # Check if the provided directory exists on local, and if not, create it
    os.makedirs(name=path, exist_ok=True)

    # Compose the local path of the input dataset
    file_path = os.path.join(path, '{0}.csv'.format(dataset))

    # If the chosen dataset is one of the standard open-access benchmark ones
    canonical = dataset in ['PJM', 'NP', 'BE', 'FR', 'DE']
    if canonical:
        # if not yet downloaded into the local input folder, then it will be downloaded.
        if not os.path.exists(file_path):
            file_url = 'https://zenodo.org/records/4624805/files/{0}.csv'.format(dataset)
            req = requests.get(url=file_url)
            with open(file=file_path, mode='wb') as f:
                f.write(req.content)

        # set index_col parameter to a preset value
        index_col = 0

        # set response_col parameter to a preset value
        if dataset == 'PJM':
            response_col = 'Zonal COMED price'
        elif dataset in ['NP', 'DE']:
            response_col = 'Price'
        elif dataset in ['BE', 'FR']:
            response_col = 'Prices'

        # set common parameters of the canonical input datasets
        sep = ','
        decimal = '.'
        date_format = 'ISO8601'
        encoding = 'utf-8'

    else:
        # if not yet downloaded into the local input folder
        if not os.path.exists(file_path):
            raise FileNotFoundError('The dataset {0}.csv is not available on {1}/'.
                                    format(dataset, os.path.abspath(path=path)))

        # Check if index_col can be valid
        if index_col is None:
            raise ValueError('Please, provide a valid index column index or name!')
        elif not isinstance(index_col, (int, str)):
            raise ValueError('The `index_col` should be either a single integer or a single string.')
        elif index_col == '':
            raise ValueError('The `index_col` should be a non-empty string (or integer). '
                             'If the index column has no name, then define it by its index position.')

        # Check if response_col can be valid
        if response_col is None:
            raise ValueError('Please, provide a valid response column index or name!')
        elif not isinstance(response_col, (int, str)):
            raise ValueError('The `response_col` should be either a single integer or a single string.')
        elif response_col == '':
            raise ValueError('The `response_col` should be a non-empty string (or integer). '
                             'If the response column has no name, then define it by its index position.')

    # read only the header of the input dataset
    orig_col_names = _detect_col_names(file_path=file_path, sep=sep, encoding=encoding, index_col=index_col)

    # Import the input dataset from local disk
    try:
        df_ = pd.read_csv(filepath_or_buffer=file_path, sep=sep, decimal=decimal, date_format=date_format,
                          encoding=encoding, index_col=index_col)
    except FileNotFoundError as error:
        raise error

    # check if the input dataset is empty
    if df_.empty:
        raise ValueError('The input dataset should not be empty.')

    # check if all columns are numeric
    if df_.shape != df_.select_dtypes(include=np.number).shape:
        print(df_.info())
        raise ValueError('The input dataset should only contain numeric values.')

    # arrange possible datetime index issues of not canonical datasets
    if not canonical:
        df_ = _arrange_index(dataset=df_, index_tz=index_tz, market_tz=market_tz, intended_freq=intended_freq)

    # sort the dataset by the index if it is not sorted yet
    if not df_.index.is_monotonic_increasing:
        df_.sort_index(inplace=True)

    # check if the input dataset has no duplicate indices
    if df_.index.has_duplicates:
        raise IndexError('The input dataset should not have duplicate indices.')

    # check if the input dataset has a consistent frequency (no missing records)
    if not df_.index.inferred_freq:
        raise IndexError('The indices of the input dataset should have a consistent frequency.')

    # detect the index of the response_col variable
    response_index = []
    if isinstance(response_col, str):
        response_index = [key for key, value in enumerate(orig_col_names) if value.strip() == response_col]
    elif isinstance(response_col, int):
        response_index = [key for key, value in enumerate(orig_col_names) if key == response_col - 1]
    if len(response_index) == 0:
        raise ValueError("The `response_col` '{0}' does not exist in the dataset.".format(response_col))

    # extract the column indices of the exogenous variables
    exogen_indices = [key for key, value in enumerate(orig_col_names) if key not in response_index]

    # rename all the columns accordingly
    df_.rename(columns={orig_col_names[i]: 'Price' for i in response_index},
               inplace=True)
    df_.rename(columns={orig_col_names[i]: 'Exogenous_{0}'.format(j + 1) for j, i in enumerate(exogen_indices)},
               inplace=True)
    df_.columns.name = None

    # set the index name
    df_.index.name = 'date'

    # if summary should be displayed
    if summary:
        # print the column renaming rules
        print("\n\t*** header renamed as *** ")
        new_col_names = df_.columns.values.tolist()
        for ind, orig_col_name in enumerate(orig_col_names):
            print("'{0}' -> '{1}'".format(orig_col_name, new_col_names[ind]))

        # show summary statistics of df_
        print("\n\t*** dataset summary *** ", df_.index._summary(),
              "inferred frequency of the index: {0}".format(df_.index.inferred_freq),
              df_.describe(include="all"), sep="\n")

    return df_


def split_data(df, years_test=2, begin_test_date=None, end_test_date=None,
               summary=False):
    """ Split the day-ahead electricity price data between training and testing dataset based
    on the related arguments provided.
     
    Parameters
    ----------
        df : pandas.DataFrame
            The dataframe which should be time-wise split to train and test set.
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
        summary : bool
            Whether to print a summary of the resulting DataFrames.

    Returns
    -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            Training dataset, testing dataset
    """
    # The training and test datasets can be defined by providing a number of years for testing
    # or by providing the start and end date of the test period
    if begin_test_date is None and end_test_date is None and years_test is not None:

        if not isinstance(years_test, (float, int)):
            raise TypeError("The years_test should be numerical object")

        if years_test <= 0:
            raise ValueError("The years_test should be positive!")

        # Pick the last complete day's last period's positional index of the whole dataset
        period_length = {'h': 60, '5min': 5, '15min': 15, '30min': 30}.get(df.index.inferred_freq)
        last_complete_days_last_index = np.where((df.index.hour == 23) &
                                                 (df.index.minute == 60 - period_length))[0][-1]

        # Choose the last complete day's last period's index (which is a timestamp)
        # of the whole dataset as the end timestamp of the test dataset
        end_test_date = df.index[last_complete_days_last_index]

        # Calculate the end timestamp of the train dataset from back to front.
        # We consider a year to be exactly 52 weeks (364 days) long, instead of the general 365 days.
        end_train_date = end_test_date - pd.Timedelta(days=math.ceil(364 * years_test))

        # Calculate the start timestamp of the test dataset.
        # This must be one period after the end timestamp of the train dataset.
        begin_test_date = df.index[df.index > end_train_date][0]

    elif begin_test_date is not None and end_test_date is not None:

        if not isinstance(begin_test_date, (pd.Timestamp, str)):
            raise TypeError("The begin_test_date should be a pandas Timestamp object"
                            " or a string with '%Y-%m-%d 00:00:00' format.")

        if not isinstance(end_test_date, (pd.Timestamp, str)):
            raise TypeError("The end_test_date should be a pandas Timestamp object"
                            " or a string with '%Y-%m-%d 23:00:00' format.")

        # Convert dates to pandas Timestamp
        begin_test_date = pd.to_datetime(arg=begin_test_date, dayfirst=False)
        end_test_date = pd.to_datetime(arg=end_test_date, dayfirst=False)

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
        raise ValueError("The test period closing hour should be 23:xx instead of {0}:{1}.".
                         format(end_test_date.hour, end_test_date.minute))

    if begin_test_date >= df.index[-1]:
        raise ValueError("The test period opening should be before the end of the input dataset.")

    if begin_test_date <= df.index[0]:
        raise ValueError("The test period opening should be after the beginning of the input dataset.")

    # Split the dataset into training and test datasets
    df_train_ = df.loc[df.index < begin_test_date, :]
    df_test_ = df.loc[begin_test_date:end_test_date, :]

    if summary:
        # show summary statistics of df_train_ and df_test_
        for df_half_name, df_half in {"df_train": df_train_, "df_test": df_test_}.items():
            print("\n\t*** {0} summary *** ".format(df_half_name), df_half.index._summary(),
                  "inferred frequency of the index: {0}".format(df_half.index.inferred_freq),
                  df_half.describe(include="all"), sep="\n")
    else:
        # show basic info of df_train_ and df_test_
        print('Training dataset period: {0} - {1}'.format(df_train_.index[0], df_train_.index[-1]))
        print('Testing dataset period: {0} - {1}'.format(df_test_.index[0], df_test_.index[-1]))

    return df_train_, df_test_


def read_and_split_data(path=os.path.join('.', 'datasets'), dataset='PJM', index_col=None, response_col=None,
                        sep=',', decimal='.', date_format='ISO8601', encoding='utf-8', years_test=2,
                        begin_test_date=None, end_test_date=None, index_tz=None, market_tz=None, intended_freq='1h',
                        summary=False):
    """ Import the day-ahead electricity price data supplemented with exogenous explanatory variables,
    and provides a split between training and testing dataset based on the related arguments provided.
    It also renames the columns of the training and testing dataset to match the requirements of the
    current module's prediction models.
    Namely, assuming that there are N exogenous explanatory variable inputs,
    the columns of the resulting dataframes will be named as
    ['Price', 'Exogenous_1', 'Exogenous_2', ..., 'Exogenous_N'].

    The returning tables' indices are naive timestamps in the format '%Y-%m-%d %H:%M:%S',
    and denote the starting timestamp of the related electricity delivery period.
    Those timestamps basically should be interpreted as the local time of the corresponding market.
    On those markets, which use Daylight Saving Time (DST), an apriori data preparation is required and done.
    When DST observation begins, the clocks are advanced by one hour during the very early morning.
    When DST observation ends and the standard time observation resumes,
    the clocks are turned back one hour during the very early morning.
    Hence, there is a missing hour in the dataset in every spring, and there is a duplicate hour in every autumn.
    The models in this module cannot handle well such missing and duplicate hours,
    so the missing hours should be imputed and the duplicate hours should be averaged in advance.

    If `dataset` is one of the online available standard market datasets of the study,
    such as "PJM", "NP", "BE", "FR" or "DE", the function checks
    whether the `XY.csv` already exists in the `path` folder.
    If not, it downloads the data_ from <https://zenodo.org/records/4624805> and saves it into there.
        - "PJM" refers to the Pennsylvania-New Jersey-Maryland day-ahead market,
        - "NP" refers to the Nord Pool day-ahead market,
        - "BE" refers to the EPEX-Belgium day-ahead market
        - "FR" refers to the EPEX-France day-ahead market
        - "DE" refers to the EPEX-Germany day-ahead market
    Note, that the data available online for these five markets is limited to certain periods.
    
    Parameters
    ----------
        path : str
            Path of the local folder where the input dataset should be placed locally.
        dataset : str
            Name of the csv file (denoted w/o "csv" extension) containing the input dataset.
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
        tuple[pandas.DataFrame, pandas.DataFrame]
            Training dataset, testing dataset
    """
    # read and prepare the input dataset
    df_ = read_data(path=path, dataset=dataset, index_col=index_col, response_col=response_col,
                    sep=sep, decimal=decimal, date_format=date_format, encoding=encoding,
                    index_tz=index_tz, market_tz=market_tz, intended_freq=intended_freq, summary=summary)

    # The training and test datasets can be defined by providing a number of years for testing
    # or by providing the start and end date of the test period
    df_train_, df_test_ = split_data(df=df_, years_test=years_test, begin_test_date=begin_test_date,
                                     end_test_date=end_test_date, summary=summary)

    return df_train_, df_test_


if __name__ == '__main__':
    # These codes below will only be executed if this script is run as the main program.
    df_train, df_test = read_and_split_data(
        path=os.path.join('..', '..', 'examples', 'datasets'),
        dataset='DE',
        # index_col=0,
        # response_col=1,
        # sep=',',
        # decimal='.',
        # date_format='ISO8601',
        # encoding='utf-8',
        years_test=2,
        # begin_test_date="2016-01-01 00:00:00",
        # end_test_date="2016-02-01 23:00:00",
        # index_tz='CET',
        # market_tz='CET',
        # intended_freq='1h',
        summary=True,
    )
    #     *** header renamed as ***
    # 'Price' -> 'Price'
    # 'Ampirion Load Forecast' -> 'Exogenous_1'
    # 'PV+Wind Forecast' -> 'Exogenous_2'
    #
    #     *** dataset summary ***
    # DatetimeIndex: 52416 entries, 2012-01-09 00:00:00 to 2017-12-31 23:00:00
    # inferred frequency of the index: h
    #               Price   Exogenous_1   Exogenous_2
    # count  52416.000000  52416.000000  52416.000000
    # mean      34.677858  21355.247119  10709.598740
    # std       15.939956   3743.920548   7643.932222
    # min     -221.990000  10718.250000    302.237500
    # 25%       26.170000  18269.250000   4545.568125
    # 50%       33.450000  21289.875000   8923.732500
    # 75%       42.582500  24452.312500  15224.571625
    # max      210.000000  35499.250000  48587.881000
    #
    #     *** df_train summary ***
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
    #     *** df_test summary ***
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
