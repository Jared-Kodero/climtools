"""
FOTRAN module abstraction layer, for functions written in FOTRAN.
This module provides a Python interface to the FOTRAN functions for trend analysis.
"""

from collections import namedtuple

import numpy as np

from .f_stats import f_stats as fs


def mk_test(arr: np.ndarray, alpha: float = 0.05, lag: int = None):
    """
    Perform the Hamed-Rao trend test on a numpy array.

    Parameters
    ----------
    arr : np.ndarray
        The input array to test for trends.
    alpha : float, optional
        Significance level for the test, default is 0.05.
    lag : int, optional
        The lag to use in the test, default is None which uses the length of the array.
    Returns
    -------
        - trend: tells the trend (1 for increasing, -1 for decreasing, 0 for no trend)
        - mean: mean of the input array
        - std: standard deviation of the input array
        - p: p-value of the significance test
        - z: normalized test statistics
        - Tau: Kendall Tau
        - slope: Theil-Sen estimator/slope

    """

    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    n = len(arr)

    if lag is None:
        lag = n
    else:
        lag = lag + 1
    slope, pval, trend, mean, std, tau, z = fs.mk_hamed_rao_test(
        arr,
        n,
        alpha,
        lag,
    )

    res = namedtuple(
        "Modified_Mann_Kendall_Test_Hamed_Rao_Approach",
        ["slope", "p", "trend", "mean", "std", "Tau", "z"],
    )

    return res(slope, pval, trend, mean, std, tau, z)
