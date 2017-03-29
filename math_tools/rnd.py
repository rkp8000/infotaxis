#!/anaconda/bin/python

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
import numpy as np
import pdb

# Random number/vector/matrix generators
def catrnd(pr_dis,shape=(1,)):
    """Return several samples from a categorical probability distribution
    
    catrnd returns several random numbers from a categorical probability 
    distribution.
    
    Args:
        pr_dis: probability distribution; 1-D array; if it does not sum to 1, 
        it will be normalized
        
        shape: tuple containing dimensions of random array to be returned; 
        default (1,)
        
    Returns:
        sample_array: array of random samples from provided categorical 
        distribution
    
    Example:
        >>> pr_dis = np.array([.2, .2, .3, .3])
        >>> catrnd(pr_dis,(2,2))
        array([[0, 1],
               [2, 2])
    """
    # Convert distribution to floating point numbers
    pr_dis_float = pr_dis.astype(float)
    # Normalize probability distribution
    pr_dis_float /= np.sum(pr_dis_float)
    # Get cumulative distribution
    cum_dis = pr_dis_float.cumsum()
    
    # Get total number of samples
    num_samples = np.prod(list(shape))
    # Make random vector of num_samples uniform random numbers
    rand_vec = np.random.random((num_samples,1))
    # Tile this vector so that it has the same number of columns as the sampl
    # space.
    rand_mat = np.tile(rand_vec,(1,pr_dis.size))
    # Get matrix where each row is a pr_dis vector for comparison with rand_mat
    cum_dis_mat = np.tile(cum_dis,(num_samples,1))
    # Create sample vector
    sample_vec = np.sum(rand_mat > cum_dis_mat,1)
    # Reshape sample vector into desired array
    sample_array = sample_vec.reshape(shape)
    
    # Just one number desired?
    if shape == (1,):
        sample_array = sample_array[0]
        
    return sample_array
    
def wishrnd(scale_mat=np.eye(2),deg_freedom=2):
    """Samples from a Wishart distribution with integer degrees of freedom
    
    wishrnd draws a sample from a wishart distribution with a given scale 
    matrix and integer degrees of freedom.
    
    Args:
        scale_mat: scale matrix; this is the covariance matrix used to draw 
        the multivariate normal samples used to construct a Wishart sample;
        symmetric and positive semidefinite 2-D array (default: np.eye(2))
        
        deg_freedom: degrees of freedom; this is the number o multivariate
        normal samples used to construct the Wishart sample; int (default: 2)
    
    Returns:
        wish_sample: sample from wishart distribution
    
    Example:
        >>> scale_mat = np.array([[3,1],[1,3]])
        >>> deg_freedom = 5
        >>> mathmagic.wishrnd(scale_mat, deg_freedom)
        array([[ 17.49065107,   1.28866693],
               [  1.28866693,   6.72369625]])
    """
    # Get dimensionality of samples
    d = scale_mat.shape[0]
    # Allocate space for storing samples of xx', where x is a sample from a
    # multivariate nomal distribution
    cov_samples = np.empty((deg_freedom, d, d))
    # Fill in samples
    for ii in np.arange(deg_freedom):
        # Draw sample from multivariate normal
        mvn_sample = np.random.multivariate_normal(np.zeros(d), scale_mat)
        # Multiply sample by its tranpose
        cov_samples[ii,:,:] = np.outer(mvn_sample, mvn_sample)
    # Sum over all the samples
    wish_sample = np.sum(cov_samples,0)
    return wish_sample
    
def iwishrnd(scale_mat=np.eye(2), deg_freedom=2):
    """Samples from an inverse wishart distribution
    
    iwishrnd returns a sample from an inverse Wishart distribution; this is a
    distribution over matrices whose inverses are Wishart distributed; it is
    useful because it is the conjugate prior of the multivariate normal 
    distribution with respect to the covariance matrix.
    
    Args:
        scale_mat: inverse Wishart scale matrix; this is the inverse of the 
        scale matrix of the Wishart distribution according to which inverses 
        of samples from this function are distributed; 2D array whose inverse
        is symmetric and positive semidefinite (default: np.eye(2))
        
        deg_freedom: degrees of freedom; see documentation for 
        mathmagic.wishrnd; int (default: 2)
        
    Returns:
        iwish_sample: sample from inverse Wishart distribution
        
    Example:
        >>> scale_mat = np.array([[3,1],[1,3]])
        >>> deg_freedom = 8
        >>> mathmagic.iwishrnd(scale_mat,deg_freedom)
        array([[ 1.4654731 , -0.21627391],
               [-0.21627391,  1.45103688]])
    """
    # Calculate inverse of scale matrix
    iscale_mat = np.linalg.inv(scale_mat)
    # Draw sample from Wishart using iscale_mat as scale matrix
    wish_sample = wishrnd(iscale_mat,deg_freedom)
    # Calculate inverse of sample from Wishart
    iwish_sample = np.linalg.inv(wish_sample)
    return iwish_sample

def ONmatrnd(dim=3):
    """ Sample from uniformly distributed orthonormal matrices.
    
    Returns an orthonormal matrix sampled from a uniform distribution over
    orthonormal rotations in N-dimensions. The algorithm operates by first
    sampling an N x N matrix of independent, normally distributed values and
    then by performing Gram-Schmidt orthogonalization on it. It has been shown
    that the resulting matrices are uniformly distributed and orthogonal.
    
    Args:
        dim: Dimension of matrix. Int.
        
    Returns:
        Orthonormal matrix sampled from uniform distribution.
        
    Example:
        >>> ONmatrnd(dim=2)
        array([[-0.13743942, -0.99051017],
               [-0.99051017,  0.13743942]])
        >>> ONmatrnd(dim=3)
        array([[-0.92156136, -0.35380929,  0.15982378],
               [-0.10942726, -0.15825923, -0.98131529],
               [ 0.37249205, -0.92183133,  0.10712921]])
    """
    
    # Sample dim x dim standard normally distributed numbers
    norm_mat = np.random.normal(0,1,(dim,dim))
    
    # Perform QR factorization.
    q,r = np.linalg.qr(norm_mat)
    
    # Return q, which is orthonormal, and uniformly distributed.
    return q
    
def mvnrnd(mu=np.array([0.,0]),K=np.array([[1.,0],[0,1]]),n=1):
    """Sample from a multivariate normal distribution.
    
    Args:
        mu: Mean vector.
        K: Covariance matrix. Must be positive semidefinite.
        n: Number of samples to return.
    """
    
    dim = len(mu)
    samples_uncor = np.random.normal(0,1,(dim,n))
    cov_sqrt = np.linalg.cholesky(K)
    samples_cor = cov_sqrt.dot(samples_uncor)
    samples_cor = (samples_cor.T + mu).T
    
    return np.squeeze(samples_cor)