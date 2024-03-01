"""
Functions to compute and plot the univariate and multivariate versions of
the one-sided Diebold-Mariano (DM) test.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl


def DM(p_real, p_pred_1, p_pred_2, norm=1, version='univariate'):
    """ Performs the one-sided Diebold-Mariano test.
    
    The test compares whether there is a difference in terms of predictive accuracy
    between the two forecasts ``p_pred_1`` and ``p_pred_2``.

    In particular, the one-sided DM test evaluates the null hypothesis versus the alternative one.
    H0 - the forecasting errors of ``p_pred_1`` are smaller or equal (better) than
    the predictive accuracy of ``p_pred_2``.
    H1 - the forecasting errors of ``p_pred_1`` are higher (worse) than the predictive accuracy of ``p_pred_2``.

    Rejecting the H0 (p-value < 5%) means that the forecast ``p_pred_2`` is significantly more accurate
    than forecast ``p_pred_1``. (Note that this is an informal definition. For a formal one we refer to 
    `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_)

    Two versions of the test are possible:

        1. A ``univariate`` version with as many independent tests performed as many prices are per day,
        i.e., 24 tests in most day-ahead electricity markets.

        2. A multivariate version with the test performed jointly for all hours using the multivariate
        loss differential series (see this `article <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_
        for details).
    
    Parameters
    ----------
        p_real : np.ndarray
            Array of shape :math:`(n_days, n_prices/day)` representing the real market
            prices
        p_pred_1 : np.ndarray
            Array of shape :math:`(n_days, n_prices/day)` representing the first forecast
        p_pred_2 : np.ndarray
            Array of shape :math:`(n_days, n_prices/day)` representing the second forecast
        norm : int
            Norm used to compute the loss differential series. At the moment, this value must either
            be 1 (np.abs(loss1) - np.abs(loss2)) or 2 (loss1**2 - loss2**2).
        version : str
            Version of the test as defined in `here <https://epftoolbox.readthedocs.io/en/latest/modules/cite.html>`_.
            It can have two values:
                - ``'univariate``
                - ``'multivariate``
    Returns
    -------
        float | np.ndarray
            The p-value(s) after performing the test. Either it is one p-value value in the case
            of the ``multivariate`` test, or a numpy.ndarray of 24 p-values for the ``univariate`` test
    """

    # Check that all time series have the same shape
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensure that time series have shape (n_days, n_prices_day)
    if p_real.ndim > 2:
        raise ValueError('The time series are {0} dimensional although they must have maximum 2 dimensions'.
                         format(p_real.ndim))

    # Compute the errors of each forecast
    loss1 = p_real - p_pred_1
    loss2 = p_real - p_pred_2

    # Compute the test statistic
    if version == 'univariate':
        # Compute the loss differential series for the test
        if norm == 1:
            d = np.abs(loss1) - np.abs(loss2)
        elif norm == 2:
            d = loss1**2 - loss2**2
        else:
            raise ValueError('The norm must be either 1 or 2')
        # Compute the loss differential size
        n = d.shape[0]

        # Compute the test statistic
        mean_d = np.mean(d, axis=0)
        var_d = np.var(d, ddof=0, axis=0)
        dm_stat = mean_d / np.sqrt(var_d / n)

    elif version == 'multivariate':
        # Compute the loss differential series for the multivariate test
        if norm == 1:
            d = np.mean(a=np.abs(loss1), axis=1) - np.mean(a=np.abs(loss2), axis=1)
        elif norm == 2:
            d = np.mean(a=loss1**2, axis=1) - np.mean(a=loss2**2, axis=1)
        else:
            raise ValueError('The norm must be either 1 or 2')

        # Compute the loss differential size
        n = d.size

        # Compute the test statistic
        mean_d = np.mean(a=d)
        var_d = np.var(d, ddof=0)
        dm_stat = mean_d / np.sqrt(var_d / n)
    
    else:
        raise ValueError('The version must be either "univariate" or "multivariate"')
        
    p_value = 1 - stats.norm.cdf(x=dm_stat)

    return p_value


def plot_multivariate_DM_test(real_price, forecasts, norm=1, title='DM test', savefig=False, path='.') -> None:
    """ Plot the comparison of forecasts using the multivariate DM test.
    
    The resulting plot is a heat map in a chessboard shape. It represents the p-value
    of the null hypothesis of the forecast in the y-axis being significantly more
    accurate than the forecast in the x-axis. In other words, p-values close to 0
    represent cases where the forecast in the x-axis is significantly more accurate
    than the forecast in the y-axis.
    
    Parameters
    ----------
        real_price : pd.DataFrame
            Dataframe that contains the real prices
        forecasts : pd.DataFrame
            Dataframe that contains the forecasts of different models. The column names are the
            forecast/model names. The number of datapoints should equal the number of datapoints
            in ``real_price``.
        norm : int
            Norm used to compute the loss differential series. At the moment, this value must either
            be 1 (for the norm-1) or 2 (for the norm-2).
        title : str
            Title of the generated plot
        savefig : bool
            Boolean that selects whether the figure should be saved into the current folder
        path : str
            Path to save the figure. Only necessary when `savefig=True`
    """
    # Compute the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elements (the same models) we directly set the p-value of 1.
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = DM(p_real=real_price.values.reshape(-1, 24), 
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1, 24), 
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1, 24), 
                                                  norm=norm,
                                                  version='multivariate')

    # Define color map
    red = np.concatenate([np.linspace(start=0, stop=1, num=50),
                          np.linspace(start=1, stop=0.5, num=50)[1:],
                          [0]])
    green = np.concatenate([np.linspace(start=0.5, stop=1, num=50),
                            np.zeros(shape=50)])
    blue = np.zeros(shape=100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1),
                                    green.reshape(-1, 1),
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generate figure
    plt.imshow(X=p_values.astype(dtype=float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts.columns)), forecasts.columns, rotation=90.)
    plt.yticks(range(len(forecasts.columns)), forecasts.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    plt.title(label=title)
    plt.tight_layout()

    if savefig:
        os.makedirs(name=path, exist_ok=True)
        plt_path = os.path.join(path, title)
        plt.savefig(plt_path + '.png', dpi=300)
        plt.savefig(plt_path + '.eps')

    plt.show()


if __name__ == '__main__':
    # These codes below will only be executed if this script is run as the main program.
    from epftoolbox.data import read_and_split_data
    import pandas as pd
    import os

    # Generate forecasts of multiple models

    # Download available day-ahead electricity price forecast of
    # the Nord Pool market available in the library repository.
    # These forecasts accompany the original paper
    forecasts = pd.read_csv('https://raw.githubusercontent.com/jeslago/epftoolbox/master/'
                            'forecasts/Forecasts_NP_DNN_LEAR_ensembles.csv', index_col=0)

    # Delete the real price field as it is the actual observed price and not a forecast
    forecasts = forecasts.drop(columns='Real price')

    # Transform indices to datetime format
    forecasts.index = pd.to_datetime(forecasts.index)

    # Read the real day-ahead electricity price data of the Nord Pool market
    # The scope period should be the same as in forecasted data.
    _, df_test = read_and_split_data(path=os.path.join('..', '..', 'examples', 'datasets'),
                                     dataset='NP',
                                     response_col='Price',
                                     begin_test_date=forecasts.index[0],
                                     end_test_date=forecasts.index[-1])
    # Training dataset period: 2013-01-01 00:00:00 - 2016-12-26 23:00:00
    # Testing dataset period: 2016-12-27 00:00:00 - 2018-12-24 23:00:00

    # Extract the real day-ahead electricity price data
    real_price = df_test.loc[:, ['Price']]

    # Calculate the univariate Diebold-Mariano test on an ensemble of DNN models versus an ensemble of LEAR models
    univ_p = DM(p_real=real_price.values.reshape(-1, 24),
                p_pred_1=forecasts.loc[:, 'LEAR Ensemble'].values.reshape(-1, 24),
                p_pred_2=forecasts.loc[:, 'DNN Ensemble'].values.reshape(-1, 24),
                norm=1,
                version='univariate')
    print('univariate DM test:', *[(i, p.round(decimals=8)) for i, p in enumerate(univ_p)], sep='\n\t')
    # univariate DM test:
    #     (0, 1.0)
    #     (1, 0.99988816)
    #     (2, 0.91413885)
    #     (3, 0.92259886)
    #     (4, 0.93470312)
    #     (5, 0.99454674)
    #     (6, 0.3636968)
    #     (7, 0.00029573)
    #     (8, 0.00031402)
    #     (9, 0.00021321)
    #     (10, 0.00753069)
    #     (11, 0.0077462)
    #     (12, 0.01318551)
    #     (13, 0.01955294)
    #     (14, 0.02259846)
    #     (15, 0.0089445)
    #     (16, 0.0005398)
    #     (17, 0.02093738)
    #     (18, 0.00603305)
    #     (19, 0.00027285)
    #     (20, 0.13607714)
    #     (21, 0.33256647)
    #     (22, 0.52448957)
    #     (23, 0.01382751)

    # Calculate the multivariate Diebold-Mariano test on an ensemble of DNN models versus an ensemble of LEAR models
    multi_p = DM(p_real=real_price.values.reshape(-1, 24),
                 p_pred_1=forecasts.loc[:, 'LEAR Ensemble'].values.reshape(-1, 24),
                 p_pred_2=forecasts.loc[:, 'DNN Ensemble'].values.reshape(-1, 24),
                 norm=1,
                 version='multivariate')
    print('multivariate DM test:', multi_p.round(decimals=8), sep='\n\t')
    # multivariate DM test:
    #     0.01409985

    # Plot the comparison of models using the multivariate DM test
    plot_multivariate_DM_test(real_price=real_price, forecasts=forecasts, norm=1,
                              title='DM test\nThe greener the area the more accurate the forecast'
                                    '\nin the x-axis than the forecast in the y-axis.',
                              savefig=False)
