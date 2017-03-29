"""
Created on Mon Jan  5 09:39:31 2015

@author: rkp

Contains some basic signal processing functions not found in scipy.
"""
from __future__ import division

import numpy as np
from numpy import concatenate as cc
from scipy.signal import fftconvolve
from scipy.stats import pearsonr
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d as smooth
import matplotlib.pyplot as plt

from math_tools import stats as mt_stats

# TEST SIGNALS
T = np.arange(300)
X = smooth((np.random.uniform(0, 1, 300) > .95).astype(float), 5)
HT = np.arange(-20, 20)
H = np.exp(-HT/7.)
H[H > 1] = 0
Y = np.convolve(X, H, mode='same')

T1 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1])


def fftxcorr(x, y, dt=1.):
    """Calculate the cross correlation between two signals using fft.
    
    Returns:
        time vector, cross correlation vector
    """
    
    # get length of signal
    sig_len = len(x)
    
    # build time vector & triangular normalization function
    if sig_len % 2:
        # If odd
        tri_norm_asc = np.arange(np.ceil(sig_len / 2.), sig_len)
        tri_norm_desc = np.arange(sig_len, np.floor(sig_len / 2.), -1)
        t = np.arange(-np.floor(sig_len / 2.), np.ceil(sig_len / 2.))
    else:
        # If even
        tri_norm_asc = np.arange(sig_len / 2., sig_len)
        tri_norm_desc = np.arange(sig_len, sig_len / 2., -1)
        t = np.arange(-np.floor(sig_len / 2.), np.ceil(sig_len / 2.))
    t *= dt
    tri_norm = np.concatenate([tri_norm_asc, tri_norm_desc])
    
    # subtract mean of signals & divide by std
    x_zero_mean = x - x.mean()
    x_clean = x_zero_mean / x_zero_mean.std()
    
    y_zero_mean = y - y.mean()
    y_clean = y_zero_mean / y_zero_mean.std()
    
    # calculate cross correlation
    xy = fftconvolve(x_clean, y_clean[::-1], mode='same')
    
    # normalize signal by triangle function
    xy /= tri_norm
    
    return t, xy


def test_fftxcorr():
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(T, X)
    axs[1].plot(T, Y)
    
    TC, XY = fftxcorr(Y, X)
    axs[2].plot(TC, XY)


def segment_by_threshold(data, th=1., seg_start='last', seg_end='next',
                         idxs=None):
    """Function to segment a 1D data time-series."""
    
    # create mask of all points that at least threshold
    above_th = (np.array(data) >= th).astype(int)
    # create onset/offset trigger array
    trigger = np.diff(cc([[0], above_th, [0]]))
    # identify onsets
    onsets = (trigger == 1).nonzero()[0]
    # identify offsets
    offsets = (trigger == -1).nonzero()[0]
    offsets -= 1
    
    # add segment starts and ends if specified
    if seg_start == 'last':
        starts = np.zeros(onsets.shape, dtype=int)
        starts[1:] = offsets[:-1] + 1
        if idxs is not None:
            starts = idxs[starts]
        starts = starts[:, None]
    else:
        starts = np.zeros(onsets.shape + (0,), dtype=int)
    if seg_end == 'next':
        ends = len(data) * np.ones(offsets.shape, dtype=int)
        ends[:-1] = onsets[1:]
        ends -= 1
        if idxs is not None:
            ends = idxs[ends]
        ends = ends[:, None]
    else:
        ends = np.zeros(onsets.shape + (0,), dtype=int)
    
    if idxs is not None:
        onsets = idxs[onsets]
        offsets = idxs[offsets]
    onsets = onsets[:, None]
    offsets = offsets[:, None]
    
    return cc([starts, onsets, offsets, ends], 1)


def test_sbt():
    starts, onsets, offsets, ends = segment_by_threshold(T1).T
    print starts
    print onsets
    print offsets
    print ends
    
    idxs = np.arange(10) + 10
    starts, onsets, offsets, ends = segment_by_threshold(T1, idxs=idxs).T
    print starts
    print onsets
    print offsets
    print ends
    

def power_spectral_density(data, fs=1.):
    """Estimate the power spectral density of a signal."""
    
    sig_lens = [len(d) for d in data]
    max_sig_len = np.max(sig_lens)
    
    psds = []
    for d in data:
        ft = np.fft.fft(d, max_sig_len)
        psds += [np.real(ft*np.conj(ft)/len(d))]
        
    f = np.arange(max_sig_len)*fs/max_sig_len
    return np.average(psds, axis=0, weights=sig_lens), f
        

def autocorrelation(data, fs=1., normed=True):
    """Estimate the autocorrelation function (and power spectral density)
    of a set of random process signals.
    
    Args:
        data: list of 1-D time-series (of potentially differing length)
        fs: sampling frequency
    Returns:
        autocorrelation function, timev ector, psd, freq vector
    """
    fs = float(fs)
    
    # autocorrelation is ft of power spectral density
    psd, f = power_spectral_density(data, fs=fs)
    acorr = np.real(np.fft.ifft(psd))
    T = len(acorr)/fs
    t = np.arange(0, T, 1/fs)
    
    if len(t) > len(acorr):
        t = t[:len(acorr)]
    
    return acorr, t, psd, f
    

def test_power_spectral_density():
    x = []
    for ii in range(10):
        n = np.random.random_integers(300,500)
        t = np.arange(n)/100.
        x += [3*np.sin(2*np.pi*30*t - np.random.normal(0,5)) + np.random.normal(0,1,t.shape)]
        
    psd, f = power_spectral_density(x, fs=100)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x[0])
    axs[0].plot(x[1])
    axs[0].plot(x[2])
    axs[1].plot(f, psd)
    

def test_autocorrelation():
    x = []
    for ii in range(10):
        n = np.random.random_integers(300,500)
        t = np.arange(n)/100.
        x += [smooth(np.random.normal(0,1,t.shape), sigma=3)]
        
    acorr, t, psd, f = autocorrelation(x, fs=100)
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(x[0])
    axs[0].plot(x[1])
    axs[0].plot(x[2])
    axs[1].plot(f, psd)
    axs[2].plot(t, acorr)


def xcov_simple_one_sided(x, y, n_lags=50, normed=False):
    """Calculate cross-covariance between time-series x and y over a certain
        number of lags. If x is white noise and y is filtered white noise, then
        the resulting cross-covariance returned will be a reconstruction of
        the filter."""

    covs = []
    for lag in range(n_lags):
        if lag == 0:
            cov = np.cov(x, y)
        else:
            # calculate the cross covariance between x and y with a specific lag
            cov = np.cov(x[:-lag], y[lag:])
        covs += [cov[0, 1]]

    covs = np.array(covs)

    if normed:
        var_x = np.var(x)
        var_y = np.var(y)
        covs /= stats.gmean([var_x, var_y])

    return covs


def xcov_simple_one_sided_multi(xs, ys, n_lags=50, normed=False):
    """
    Calculate cross-covariance between x and y when multiple time-series are available.
        This function is to be used when it is believed that y is created by filtering x
        with a causal linear filter. If that is the case, then as the number of samples
        increases, the result will approach the shape of the original filter.
    :param xs: list of input time-series
    :param ys: list of output time-series
    :param n_lags: number of lags
    :param normed: if True, results will be normalized by geometric mean of x's & y's variances
    :return: cross-covariance, p-value, lower bound, upper bound
    """

    if not np.all([len(x) == len(y) for x, y in zip(xs, ys)]):
        raise ValueError('Arrays within xs and ys must all be of the same size!')

    covs = []
    p_values = []
    lbs = []
    ubs = []

    for lag in range(n_lags):
        if lag == 0:
            cov = np.cov(np.concatenate(xs), np.concatenate(ys))
            all_xs = np.concatenate(xs)
            all_ys = np.concatenate(ys)
        else:
            # calculate the cross covariance between x and y with a specific lag
            # first get the relevant xs and ys from each time-series
            x_rel = [x[:-lag] for x in xs if len(x) > lag]
            y_rel = [y[lag:] for y in ys if len(y) > lag]
            all_xs = np.concatenate(x_rel)
            all_ys = np.concatenate(y_rel)

        cov = np.cov(all_xs, all_ys)
        rho, p_value, lb, ub = mt_stats.pearsonr_with_confidence(all_xs, all_ys)

        covs += [cov[0, 1]]
        p_values += [p_value]

        lbs += [lb]
        ubs += [ub]

    covs = np.array(covs)
    p_values = np.array(p_values)
    lbs = np.array(lbs)
    ubs = np.array(ubs)

    var_x = np.var(np.concatenate(xs))
    var_y = np.var(np.concatenate(ys))
    norm_factor = stats.gmean([var_x, var_y])

    if normed:
        # normalize by mean variance of signals
        covs /= norm_factor
    else:
        # convert confidence bounds to covariances
        lbs *= norm_factor
        ubs *= norm_factor

    return covs, p_values, lbs, ubs


def xcov_simple_two_sided_multi(xs, ys, n_lags_forward=50, n_lags_back=10, confidence=0.95, normed=False):
    """
    Calculate cross-covariance between x and y when multiple time-series are available.
        This function is to be used when it is believed that y is created by filtering x
        with a causal linear filter. If that is the case, then as the number of samples
        increases, the result will approach the shape of the original filter.
    :param xs: list of input time-series
    :param ys: list of output time-series
    :param n_lags_forward: number of lags to look forward (causal)
    :param n_lags_back: number of lags to look back (acausal)
    :param confidence: confidence of confidence interval desired
    :param normed: if True, results will be normalized by geometric mean of x's & y's variances
    :return: cross-covariance, p-value, lower bound, upper bound
    """

    if not np.all([len(x) == len(y) for x, y in zip(xs, ys)]):
        raise ValueError('Arrays within xs and ys must all be of the same size!')

    covs = []
    p_values = []
    lbs = []
    ubs = []

    for lag in range(-n_lags_back, n_lags_forward):
        if lag < 0:
            x_rel = [x[-lag:] for x in xs if len(x) > -lag]
            y_rel = [y[:lag] for y in ys if len(y) > -lag]
            all_xs = np.concatenate(x_rel)
            all_ys = np.concatenate(y_rel)
        elif lag == 0:
            all_xs = np.concatenate(xs)
            all_ys = np.concatenate(ys)
        else:
            # calculate the cross covariance between x and y with a specific lag
            # first get the relevant xs and ys from each time-series
            x_rel = [x[:-lag] for x in xs if len(x) > lag]
            y_rel = [y[lag:] for y in ys if len(y) > lag]
            all_xs = np.concatenate(x_rel)
            all_ys = np.concatenate(y_rel)

        cov = np.cov(all_xs, all_ys)
        rho, p_value, lb, ub = mt_stats.pearsonr_with_confidence(all_xs, all_ys, confidence)

        covs += [cov[0, 1]]
        p_values += [p_value]

        lbs += [lb]
        ubs += [ub]

    covs = np.array(covs)
    p_values = np.array(p_values)
    lbs = np.array(lbs)
    ubs = np.array(ubs)

    var_x = np.var(np.concatenate(xs))
    var_y = np.var(np.concatenate(ys))
    norm_factor = stats.gmean([var_x, var_y])

    if normed:
        # normalize by mean variance of signals
        covs /= norm_factor
    else:
        # convert confidence bounds to covariances
        lbs *= norm_factor
        ubs *= norm_factor

    return covs, p_values, lbs, ubs


def unmod(x, range=1, cross_over_threshold=0.3):
    """
    "Unmod" a time-series, i.e., it the last operation used to create x was
    x = mod(x, n) + b, then this function undoes that operation.

    :param x: time-series of interest
    :param range: maximum range over which this signal varies
    :param cross_over_threshold: (between 0 and 1) proportion of range between min and max
    such that if x jumps by that much it is assumed that it was actually crossing over due to the
    mod function.
    :return: A new time-series that is not constrained by "min" and "max".
    """

    # calculate non-relative threshold
    threshold = cross_over_threshold * range

    diff = np.diff(x)

    triggers = np.zeros(diff.shape, dtype=int)
    triggers[diff > threshold] = -1  # where signal jumped up due to mod
    triggers[diff < -threshold] = 1  # where signal jumped down due to mod

    n_offsets = triggers.cumsum()  # calculate how many ranges each time point should be offset by
    n_offsets = np.concatenate([[0], n_offsets])

    # return fixed signal
    return x + n_offsets * range