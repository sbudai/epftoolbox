"""
Functions to compute and plot the univariate and multivariate versions of
the one-sided Giacomini-White (GW) test for Conditional Predictive Ability
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt


def GW(p_real, p_pred_1, p_pred_2, norm=1, version='univariate'):
    """ Perform the one-sided Giacomini-White test.
    
    The test compares whether there is a difference in terms of Conditional Predictive Accuracy
    between the two forecasts ``p_pred_1`` and ``p_pred_2``.

    In particular, the one-sided GW test evaluates the null hypothesis versus the alternative one.
    H0 - the CPA errors of ``p_pred_1`` are higher or equal (better) than the CPA of ``p_pred_2``.
    H1 - the CPA errors of ``p_pred_1`` are smaller (worse) than the CPA of ``p_pred_2``.
    
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
        p_real : numpy.ndarray
            Array of shape :math:`(n_days, n_prices/day)` representing the real market
            prices
        p_pred_1 : numpy.ndarray
            Array of shape :math:`(n_days, n_prices/day)` representing the first forecast
        p_pred_2 : numpy.ndarray
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

    # This test is only implemented for a single-step forecasts
    tau = 1

    # Compute the loss differential series for the test
    if norm == 1:
        d = np.abs(loss1) - np.abs(loss2)
    elif norm == 2:
        d = loss1**2 - loss2**2
    else:
        raise ValueError('The norm must be 1 (np.abs(loss1) - np.abs(loss2)) or 2 (loss1**2 - loss2**2)')

    tt = np.max(a=d.shape)

    # Compute the Conditional Predictive Ability test statistic
    if version == 'univariate':
        gw_stat = np.inf * np.ones(shape=(np.min(a=d.shape), ))
        for h in range(24):
            instruments = np.stack(arrays=[np.ones_like(a=d[:-tau, h]), d[:-tau, h]])
            dh = d[tau:, h]
            big_t = tt - tau
            
            instruments = np.array(instruments, ndmin=2)

            reg = np.ones_like(a=instruments) * -999
            for jj in range(instruments.shape[0]):
                reg[jj, :] = instruments[jj, :] * dh
        
            if tau == 1:
                betas = np.linalg.lstsq(a=reg.T, b=np.ones(shape=big_t), rcond=None)[0]
                err = np.ones(shape=(big_t, 1)) - np.dot(a=reg.T, b=betas)
                r2 = 1 - np.mean(a=err**2)
                gw_stat[h] = big_t * r2
            else:
                raise NotImplementedError('Only one step forecasts are implemented')
    elif version == 'multivariate':
        d = d.mean(axis=1)
        instruments = np.stack(arrays=[np.ones_like(d[:-tau]), d[:-tau]])
        d = d[tau:]
        big_t = tt - tau
        
        instruments = np.array(instruments, ndmin=2)

        reg = np.ones_like(a=instruments) * -999
        for jj in range(instruments.shape[0]):
            reg[jj, :] = instruments[jj, :] * d
    
        if tau == 1:
            betas = np.linalg.lstsq(a=reg.T, b=np.ones(big_t), rcond=None)[0]
            err = np.ones(shape=(big_t, 1)) - np.dot(a=reg.T, b=betas)
            r2 = 1 - np.mean(a=err**2)
            gw_stat = big_t * r2
        else:
            raise NotImplementedError('Only one step forecasts are implemented')
    else:
        raise ValueError('The version must be "univariate" or "multivariate"')
    
    gw_stat *= np.sign(np.mean(a=d, axis=0))
    q = reg.shape[0]
    p_value = 1 - stats.chi2.cdf(gw_stat, q)

    return p_value


def plot_multivariate_GW_test(real_price, forecasts, norm=1, title='GW test', savefig=False, path='.') -> None:
    """ Plot the comparison of forecasts using the multivariate GW test.
    
    The resulting plot is a heat map in a chessboard shape. It represents the p-value
    of the null hypothesis of the forecast in the y-axis being significantly more
    accurate than the forecast in the x-axis. In other words, p-values close to 0
    represent cases where the forecast in the x-axis is significantly more accurate
    than the forecast in the y-axis.
    
    Parameters
    ----------
        real_price : pandas.DataFrame
            Dataframe that contains the real prices
        forecasts : pandas.DataFrame
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
    # Compute the multivariate GW test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elements (the same models) we directly set the p-value of 1.
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = GW(p_real=real_price.values.reshape(-1, 24), 
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
    plt.xticks(ticks=range(len(forecasts.columns)), labels=forecasts.columns, rotation=90.)
    plt.yticks(ticks=range(len(forecasts.columns)), labels=forecasts.columns)
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
                                     response='Price',
                                     begin_test_date=forecasts.index[0],
                                     end_test_date=forecasts.index[-1])
    # Training dataset period: 2013-01-01 00:00:00 - 2016-12-26 23:00:00
    # Testing dataset period: 2016-12-27 00:00:00 - 2018-12-24 23:00:00

    # Extract the real day-ahead electricity price data
    real_price = df_test.loc[:, ['Price']]

    # Calculate the univariate Giacomini-White test on an ensemble of DNN models versus an ensemble of LEAR models
    univ_p = GW(p_real=real_price.values.reshape(-1, 24),
                p_pred_1=forecasts.loc[:, 'LEAR Ensemble'].values.reshape(-1, 24),
                p_pred_2=forecasts.loc[:, 'DNN Ensemble'].values.reshape(-1, 24),
                norm=1,
                version='univariate')
    print('univariate GW test:', *[(i, p.round(decimals=8)) for i, p in enumerate(univ_p)], sep='\n\t')
    # univariate GW test:
    #     (0, 1.0)
    #     (1, 1.0)
    #     (2, 1.0)
    #     (3, 1.0)
    #     (4, 1.0)
    #     (5, 1.0)
    #     (6, 0.8658969)
    #     (7, 0.00293152)
    #     (8, 0.00291989)
    #     (9, 0.00038448)
    #     (10, 0.03982447)
    #     (11, 0.03989027)
    #     (12, 0.08743785)
    #     (13, 0.07370347)
    #     (14, 0.12498961)
    #     (15, 0.06264695)
    #     (16, 0.00512338)
    #     (17, 0.02556893)
    #     (18, 0.01988231)
    #     (19, 0.00076182)
    #     (20, 0.0341112)
    #     (21, 0.0738448)
    #     (22, 1.0)
    #     (23, 0.07383338)

    # Calculate the multivariate Giacomini-White test
    multi_p = GW(p_real=real_price.values.reshape(-1, 24),
                 p_pred_1=forecasts.loc[:, 'LEAR Ensemble'].values.reshape(-1, 24),
                 p_pred_2=forecasts.loc[:, 'DNN Ensemble'].values.reshape(-1, 24),
                 norm=1,
                 version='multivariate')
    print('multivariate GW test:', multi_p.round(decimals=8), sep='\n\t')
    # multivariate GW test:
    #     0.08663576

    # Plot the comparison of models using the multivariate DM test
    plot_multivariate_GW_test(real_price=real_price, forecasts=forecasts, norm=1,
                              title='GW test\nThe greener the area the more accurate the forecast'
                                    '\nin the x-axis than the forecast in the y-axis.',
                              savefig=False)
