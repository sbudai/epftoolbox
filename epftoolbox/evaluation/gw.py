"""
Functions to compute and plot the univariate and multivariate versions of the Giacomini-White (GW) test
for Conditional Predictive Ability
"""

import numpy as np
from scipy import stats


def gwtest(loss1: numpy.ndarray, loss2: numpy.ndarray, tau: int = 1, conditional: int = 1) -> float | np.ndarray:
    d = loss1 - loss2
    TT = np.max(d.shape)

    if conditional:
        instruments = np.stack(arrays=[np.ones_like(a=d[:-tau]), d[:-tau]])
        d = d[tau:]
        T = TT - tau
    else:
        instruments = np.ones_like(a=d)
        T = TT
    
    instruments = np.array(instruments, ndmin=2)

    reg = np.ones_like(a=instruments) * -999
    for jj in range(instruments.shape[0]):
        reg[jj, :] = instruments[jj, :] * d
    
    if tau == 1:
        # print(reg.shape, T)
        # print(reg.T)
        betas = np.linalg.lstsq(a=reg.T, b=np.ones(T), rcond=None)[0]
        # print(np.dot(reg.T, betas).shape)
        err = np.ones(shape=(T, 1)) - np.dot(a=reg.T, b=betas)
        r2 = 1 - np.mean(a=err**2)
        GWstat = T * r2
    else:
        raise NotImplementedError
        zbar = np.mean(a=reg, axis=-1)
        nlags = tau - 1
        # ...
    
    GWstat *= np.sign(np.mean(d))
    # p_value = 1 - stats.norm.cdf(GWstat)
    # if np.isnan(p_value) or p_value > .1:
    #     p_value = .1
    # return p_value
    
    q = reg.shape[0]
    p_value = 1 - stats.chi2.cdf(GWstat, q)
    # if np.isnan(p_value) or p_value > .1:
    #     p_value = .1
    return p_value
