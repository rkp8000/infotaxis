#!/anaconda/bin/python
import pdb
"""
Modified Jan 31 2013
Modified Jan 7 2013
Modified Dec 11 2013

Created Nov 29 2013

@author: R. K. Pang

This module provides a set of very useful functions, including nonstandard 
random number generators and special plotting functions.
"""

# Most of these functions rely on mathematical tools in numpy
import scipy.io as io
import scipy.stats as stats
import numpy as np

import mathmagic.fun as mmf

def mvnpdf(x,mu,K):
    """Multivariate normal probability density.
    
    Args:
        x: Vector or array of vectors.
        
        mu: Mean of distribution.
        
        K: Covariance of distribution.
        
    Returns:
        Array of multivariate probability densities, one for each row of x.
    
    Example:
        >>> mu = np.array([1,1])
        >>> K = np.array([[1,.5],[.5,2]])
        >>> x = np.array([0,0])
        >>> mvnpdf(x,mu,K)
        0.067941140344700182
        >>> x = np.array([[0,0],[1,3]])
        >>> mvnpdf(x,mu,K)
        array([ 0.06794114,  0.03836759])
    """
    
    if len(x.shape) == 1:
        is_vec = True
        x = x.reshape((1,x.shape[0]))
    else:
        is_vec = False
    
    dim = x.shape[1]
    
    K_inv = np.linalg.inv(K)
    
    # Subtract mean from x
    x_n = x - mu.squeeze()
    # Calculate stuff in exponential
    in_exp = -.5*(np.sum(x_n.T*np.dot(K_inv,x_n.T),0))
    # Calculate the normalization constant
    norm_const = (((2*np.pi)**dim)*np.abs(np.linalg.det(K)))**.5
    
    if is_vec:
        return (np.exp(in_exp)/norm_const)[0]
    else:
        return np.exp(in_exp)/norm_const

def logmvnpdf(x, mu, K, logdetK=None, opt1='standard'):
    """Calculate the log multivariate normal probability density at x.
    
    logmvnpdf calculates the natural logarithm of the probability density of 
    the samples contained in x.
    
    Args:
        x: Samples to calculate probability for. x can be given as a single 
        vector (1-D array), or as a matrix ((n x d) array). In the latter case
        if mu is a matrix ((n x d) array), and K is a (n x d x d) array, the 
        probability of the i-th row of x will be calculated under the multi-
        variate normal with its mean given by the i-th row of mu, and its 
        covariance given by the i-th plane of K.
        
        mu: Mean of distribution.
        
        K: Covariance matrix of multivariate normal distribution.
        
        logdetK: Natural log of determinant(s) of K. Float (if only one K) or 
        (n x 1) float array (if several K)
        
        opt1: Method of interpreting K. If set to 'standard', K will be 
        interpreted as the standard covariance matrix. If set to 'inverse', K 
        will be interpreted as the inverse covariance matrix.
    
    Returns:
        logprob: the logarithm of the probability density of the samples under
        the given distribution(s). Length (n) float array.
        
    Example call:
        >>> # Calculate probability of one sample under one distribution
        >>> x = np.array([1.,2.,3.,5.])
        >>> mu = np.array([0.])
        >>> K = np.array([3.])
        >>> logmvnpdf(x,mu,K)
        -3.3871832107433999
        
        >>> # Calculate probability of one sample under one distribution
        >>> x = np.array([1.,2.])
        >>> mu = np.array([0.,0.])
        >>> K = np.array([[2.,1.],[1.,2.]])
        >>> logmvnpdf(x,mu,K)
        -3.3871832107433999
        
        >>> # Calculate probabiliy of three samples under one distribution
        >>> x = np.array([[1.,2.],[0,0],[-1,-2]])
        >>> mu = np.array([0.,0.])
        >>> K = np.array([[2,1],[1,2]])
        >>> logmvnpdf(x,mu,K)
        array([-3.38718321, -2.38718321, -3.38718321])
        
        >>> # Calculate probability of three samples with three different means
        >>> # and one covariance matrix
        >>> x = np.array([[1.,2.],[0,0],[-1,-2]])
        >>> mu = np.array([[0.,0.],[1,1],[2,2]])
        >>> x -= mu
        >>> mu = np.array([0.,0])
        >>> K = np.array([[2,1],[1,2]])
        >>> logmvnpdf(x,mu,K)
        array([-3.38718321, -2.72051654, -6.72051654])
    """
    
    # If K is one-dimensional, just calculate normpdf of all samples
    if K.size == 1:
        if not (isinstance(mu,int) or isinstance(mu,float)):
            mu = mu.item(0)
        if not (isinstance(K,int) or isinstance(K,float)):
            K = K.item(0)
        return -.5*np.log(2*np.pi*K) - .5*((x-mu)**2)/K
        
    # Remove extraneous dimension from x and mu
    x = np.squeeze(x)
    mu = np.squeeze(mu)
    z = (x - mu).T
    # Make sure there are as many samples as covariance matrices and figure out
    # how many total calculations we'll need to do

    # Calculate inverses and log-determinants if necessary
    if not opt1.lower() == 'inverse':
        # Have multiple covariance matrices been supplied?
        if len(K.shape) == 3:
            # Calculate inverses
            Kinv = np.linalg.inv(K)
            # Calculate log determinants
            if logdetK is None:
                logdetK = np.log(np.linalg.det(K))
        else:
            # Calculate inverse
            Kinv = np.linalg.inv(K)
            # Calculate log determinant
            if logdetK is None:
                logdetK = np.log(np.linalg.det(K))
    else:
        Kinv = K.copy()
        # Have log-determinants been provided?
        if logdetK is None:
            # Multiple covariance matrices?
            K = np.linalg.inv(K)
            detK = np.det(K)
            logdetK = np.log(detK)

    # Calculate matrix product of z*Kinv*z.T for each Kinv and store it in y.
    temp1 = np.dot(Kinv,z)
    if len(z.shape) == 2:
        mat_prod = (z*temp1).sum(0)
    else:
        mat_prod = np.dot(z,temp1)

    # Get dimension of system
    dim = z.shape[0]
    
    # Calculate final log probability
    logprob = -.5*(dim*np.log(2*np.pi) + logdetK + mat_prod)
    
    # Remove extraneous dimension
    return np.squeeze(logprob)