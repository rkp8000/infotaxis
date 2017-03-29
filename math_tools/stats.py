"""
Modified Apr 29 2014
Modified Jan 31 2014
Modified Jan 7 2014
Modified Dec 11 2013

Created Nov 29 2013

@author: R. K. Pang

This module provides a set of statistical tools.
"""

import numpy as np
from numpy import concatenate as cc
from scipy import interpolate
import scipy.stats as stats
import statsmodels.api as sm


class dhist():
    """Dynamic histogram.
    
    Example:
    >>> x0 = [1, 7, 6, 8, 9]
    >>> x1 = [2, 3, 1, 7, 9, 8, 7, 7]
    >>> x2 = x0 + x1
    >>> bins = np.array([0,5,10])
    >>> cts, _ = np.histogram(x2, bins)
    >>> cts
    >>> dh = dhist(bins=bins)
    >>> dh.add(x0)
    >>> dh.add(x1)
    >>> dh.cts
    """
    
    def __init__(self, bins=10):
        self.bins = bins
        self.bin_centers = None
        self.count = 0
        self.mean = None
        
    def add(self, data):
        """Add data to histogram and recalculate distribution and mean."""
        
        # If first dataset
        if self.count == 0:
            
            # Compute mean
            self.mean = np.mean(data)
            # Compute counts & bins
            self.cts, self.bins = np.histogram(data, self.bins)
            # Get bin center & width
            self.bin_centers = .5 * (self.bins[:-1] + self.bins[1:])
            self.bin_width = self.bins[1] - self.bins[0]
            
        # If subsequent dataset
        else:
            
            # Update mean
            total_count = float(self.count + len(data))
            old_weight = self.count / total_count
            new_weight = len(data) / total_count
            self.mean = old_weight * self.mean + new_weight * np.mean(data)
            # Add new counts
            self.cts += np.histogram(data, self.bins)[0]
            
        # Update total count
        self.count += len(data)
        
        # Compute normalized counts
        self.normed_cts = self.cts / float(self.count)


def DKL(P,Q,symmetric=True):
    """Compute the Kullback-Liebler divergence between two probability
    distributions."""
    
    dx = 1./P.sum()
    
    # Check to make sure that Q_i = 0 ==> P_i = 0
    if np.any(P[Q==0.]):
        return np.nan
    if symmetric:
        if np.any(Q[P==0.]):
            return np.nan
    
    # Calculate log
    L1 = np.log(P/Q)
    # Set inf's to zero, because they will be zero anyhow
    L1[np.isinf(L1)] = 0.
    DKL1 = (L1*P).sum()
    
    if symmetric:
        L2 = np.log(Q/P)
        L2[np.isinf(L2)] = 0.
        DKL2 = (L2*Q).sum()
        DKL = (DKL1 + DKL2)/2.
    else:
        DKL = DKL1
        
    DKL *= dx
        
    return DKL
    
    
def lin_fit(predictors, predicted):
    """Fit a linear model to some data. Return the coefficients and the root mean 
    squared error."""
    
    if len(predicted.shape) == 1:
        predicteds = predicted[:,None]
    else:
        predicteds = predicted.copy()
        
    if predictors is None:
        predictors = np.zeros((len(predicted), 0))
        
    # add constant to predictors
    predictors_cnst = sm.add_constant(predictors)
    
    # create space for coefficients and mses (first row is for constant term)
    coeffs = np.zeros((predictors_cnst.shape[1], predicteds.shape[1]))
    mses = np.zeros((predicteds.shape[1],))
    
    # fit model for each prediction
    for pctr, data in enumerate(predicteds.T):
        model = sm.OLS(data, predictors_cnst)
        results = model.fit()
        coeffs[:,pctr] = results.params
        prediction = model.predict(results.params, predictors_cnst)
        mses[pctr] = np.mean((prediction - data)**2)
        
    return coeffs, mses


def pearsonr_with_confidence(x, y, confidence=0.95):
    """Calculate the pearson correlation coefficient, its p-value, and upper and
    lower 95% confidence bound."""

    rho, p = stats.pearsonr(x, y)
    n = len(x)

    # calculate confidence interval on correlation
    # how confident do we want to be?
    n_sds = stats.norm.ppf(1 - (1 - confidence) / 2)
    z = 0.5 * np.log((1 + rho) / (1 - rho))  # convert to z-space
    sd = np.sqrt(1. / (n - 3))
    lb_z = z - n_sds * sd
    ub_z = z + n_sds * sd
    # convert back to rho-space
    lb = (np.exp(2*lb_z) - 1) / (np.exp(2*lb_z) + 1)
    ub = (np.exp(2*ub_z) - 1) / (np.exp(2*ub_z) + 1)

    return rho, p, lb, ub


def binomial_confidence_conjugate_prior(k, n, alpha=0.05, resolution=10000):
    """
    Calculate the confidence interval for the true probability p* of a binomial distribution
    being k/n, given k counts over n observations.

    This calculation is done by assuming a max-entropy prior distribution over p* and then updating
    it in a Bayesian way and determining the edges of the resulting posterior distribution.

    :param k: number of counts
    :param n: number of observations
    :param alpha: confidence interval boundary
    :param resolution: number of points to split up 0 - 1 interval into when approximating distribution
    :return: lower bound on confidence, upper bound on confidence
    """

    # create normalized probability distribution over p*
    p = np.linspace(0, 1, resolution)
    posterior = stats.binom.pmf(k, n, p)
    posterior /= posterior.sum()

    cdf = posterior.cumsum()

    # get bounds
    lb = p[np.argmax(cdf > alpha/2) - 1]
    ub = p[-(np.argmax(cdf[::-1] < (1 - alpha/2)) - 1)]

    return lb, ub