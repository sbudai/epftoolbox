"""
Function to compute and plot the univariate and multivariate versions of the Giacomini-White (GW) test
for Conditional Predictive Ability
"""

import numpy as np
from scipy import stats


def gwtest(loss1: np.ndarray, loss2: np.ndarray, tau: int = 1, conditional: int = 1) -> float | np.ndarray:
    d = loss1 - loss2
    tt = np.max(d.shape)

    if conditional:
        instruments = np.stack(arrays=[np.ones_like(a=d[:-tau]), d[:-tau]])
        d = d[tau:]
        big_t = tt - tau
    else:
        instruments = np.ones_like(a=d)
        big_t = tt
    
    instruments = np.array(instruments, ndmin=2)

    reg = np.ones_like(a=instruments) * -999
    for jj in range(instruments.shape[0]):
        reg[jj, :] = instruments[jj, :] * d
    
    if tau == 1:
        # print(reg.shape, big_t)
        # print(reg.T)
        betas = np.linalg.lstsq(a=reg.T, b=np.ones(big_t), rcond=None)[0]
        # print(np.dot(reg.T, betas).shape)
        err = np.ones(shape=(big_t, 1)) - np.dot(a=reg.T, b=betas)
        r2 = 1 - np.mean(a=err**2)
        gw_stat = big_t * r2
    else:
        raise NotImplementedError
        # zbar = np.mean(a=reg, axis=-1)
        # nlags = tau - 1
        # ...
    
    gw_stat *= np.sign(np.mean(d))
    # p_value = 1 - stats.norm.cdf(gw_stat)
    # if np.isnan(p_value) or p_value > .1:
    #     p_value = .1
    # return p_value
    
    q = reg.shape[0]
    p_value = 1 - stats.chi2.cdf(gw_stat, q)
    # if np.isnan(p_value) or p_value > .1:
    #     p_value = .1
    return p_value
