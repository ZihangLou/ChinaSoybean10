import numpy as np
from scipy.signal import savgol_coeffs
from scipy.ndimage import convolve1d


COEFFS_LONG_TREND = savgol_coeffs(17, 2)  #
COEFFS_SHORT_TREND = savgol_coeffs(17, 6)  # 8, 6


def chen_sg_filter(curve_0, max_iteration=20):
    '''
    A simplified version of the Chen SG filter for vegetation index timeseries smooth.
    Please refer to the article for details:
        A simple method for reconstructing a high-quality NDVI time-series data set based on the Savitzky-Golay filter.
    Note:
        This algorithm is just an simplified python implementation of that article, just to demonstrate how to use the
        SMFS algorithm. The complete Chen SG filtering algorithm can be downloaded from this website:
        http://www.chen-lab.club/?post_type=products&page_id=14968
    :return:
    '''
    curve_tr = convolve1d(curve_0, COEFFS_LONG_TREND, mode="wrap")
    d = curve_tr - curve_0
    dmax = np.max(np.abs(d))
    w_func = np.frompyfunc(lambda d_i: min((1, 1 - d_i/dmax)), 1, 1)
    W = w_func(d)
    curve_k = np.copy(curve_tr)
    f_arr = np.zeros(max_iteration)
    curve_previous = None
    for i in range(max_iteration):
        curve_k = np.maximum(curve_k, curve_0)
        curve_k = convolve1d(curve_k, COEFFS_SHORT_TREND, mode="wrap")
        f_arr[i] = np.sum(np.abs(curve_k - curve_0) * W)
        if i >= 1 and f_arr[i] > f_arr[i - 1]:
            return curve_previous
        curve_previous = curve_k
    return curve_previous
