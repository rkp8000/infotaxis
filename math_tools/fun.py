import pdb
"""
Modified Apr 29 2014
Modified Jan 31 2014
Modified Jan 7 2014
Modified Dec 11 2013

Created Nov 29 2013

@author: R. K. Pang

This module provides a set of very useful functions, including nonstandard 
random number generators and special plotting functions.
"""

# Import numpy
import numpy as np
from scipy import interpolate
import scipy.stats as stats

def mwfun(func, *args):
    """Performs a function operation matrix-wise on numpy ndarray objects
    
    mwfun performs the function func on each pair of matrices in the 3D numpy
    arrays provided as arguments. All outputs must have same dimensionality, 
    e.g., they must all be c x d matrices, all be scalars, etc. 
    
    The output vals[ii] is equal to func(args[0][ii],args[1][ii],...). 
    See examples.
    
    Args:
        func: Function object.
        
        *args: Arguments to be passed matrix-wise to func. (n x m x p) numpy
        arrays, where the function operates on matrices args[ii][jj].
        
    Returns:
        vals: output of function. Length (n) 1D array if scalar ouput, (n x c)
        array if vector output. (n x c x d) if matrix output, etc.

    Examples:
        >>> K = np.array([[[3,1],[1,3]],[[5,1],[1,5]]])
        >>> # Calculate determinants
        >>> mwfun(np.linalg.det,K)
        array([8., 24.])
        >>> # Calculate inverses
        >>> mwfun(np.linalg.inv,K)
        array([[[ 0.375     , -0.125     ],
                [-0.125     ,  0.375     ]],

               [[ 0.20833333, -0.04166667],
                [-0.04166667,  0.20833333]]])
        >>> # Calculate matrix products
        >>> x = np.array([[1,1],[2,2]])
        >>> mwfun(np.dot,K,x)
        array([[  4.,   4.],
               [ 12.,  12.]])
    
    """
    # Get number of arguments
    num_args = len(args)
    # Make list of first matrices
    first_mat_list = [args[ii][0] for ii in range(num_args)]
    # Calculate function value of first matrices
    first_val = func(*first_mat_list)
    # Get number of matrices in each np.ndarray argument
    num_mats = args[0].shape[0]    
    
    # Allocate space for all function values
    if not isinstance(first_val,np.ndarray):
        vals = np.empty((num_mats,),dtype=float)
    elif len(first_val.shape) == 1:
        vals = np.empty((num_mats,first_val.shape[0]),dtype=float)
    else:
        vals = np.empty((num_mats,first_val.shape[0],first_val.shape[1]),
                        dtype=float)
    
    # Store first_val in vals
    vals[0] = first_val
    # Run through and calculate function values
    for jj in range(1,num_mats):
        # Get list of ii-th matrices
        jj_mat_list = [args[ii][jj] for ii in range(num_args)]
        # Calculate function value
        vals[jj] = func(*jj_mat_list)
        
    return vals

def statdist(tr_mat):
    """Calculates the stationary distribution of a transition matrix.
    
    statdist returns the stationary distribution of a (right stochastic) 
    transition matrix. This is the left eigenvector of the transition matrix 
    corresponding to the eigenvalue 1
    
    Args:
        tr_mat: transition matrix; right stochastic matrix, i.e., each row 
        should sum to 1; square 2D array
    
    Returns:
        statdistvec: stationary distribution vector; 1D array which sums to 1
    
    Example:
        >>> tr_mat = np.array([[.3, .4, .3],[.2, .7, .1],[.1, .1, .8]])
        >>> statdist(tr_mat)
        array([ 0.17241379,  0.37931034,  0.44827586])
    """
    tr_copy = tr_mat.copy()
    # Set any rows of Nans to equal probabilities
    tr_copy[np.isnan(tr_mat)] = 1./tr_copy.shape[0]
    # Calculate left eigenvectors and eigenvalues of transition matrix
    evs,evecs = np.linalg.eig(tr_copy.transpose())
    # Get eigenvector with eigenvalue 1
    statdistvec = evecs[:,np.argmin(np.abs(evs-1))]
    # Normalize stationary distribution and remove any spurious imaginary part
    statdistvec = np.real(statdistvec/np.sum(statdistvec))
    return statdistvec
    
def log(x):
    """Calculates logarithm, returning -inf for zero-valued elements.
    
    """
    y = np.empty(x.shape)
    y[x == 0] = -np.inf
    y[x > 0] = np.log(x[x > 0])
    return y
    
def logsum(logx):
    """Efficiently calculates the logarithm of a sum from the logarithm of its
    terms.
    
    logsum employes an efficient algorithm for finding the logarithm of a sum
    of numbers when provided with the logarithm of the terms in the sum. This 
    is especially useful when the logarithms are exceptionally large or small
    and would cause numerical errors if exponentiated.
    
    Args:
        logx: Logarithms of terms to sum. n-length float array.
    
    Returns:
        logS: Logarithm of the sum. int.
    
    Example:
        >>> logx = np.array([-1000,-1001,-1002])
        >>> np.log(np.sum(np.exp(logx)))
        -inf
        >>> logsum(logx)
        -999.59239403555557
        >>> logx = np.array([1000,1001,1002])
        >>> np.log(np.sum(np.exp(logx)))
        inf
        >>> logsum(logx)
        1002.4076059644444
    """   
    # Get largest element
    maxlogx = np.max(logx)
    # Normalize logx by subtracting maxlogx
    logx_new = logx - maxlogx
    # Calculate sum of logarithms
    logS = maxlogx + np.log(np.sum(np.exp(logx_new)))
    return logS
        
def detrend(x, mean_abs_norm=1.):
    """ Subtract mean and normalize absolute value of list of arrays.
    
    Subtract the overall mean (dimension-wise) from a 2-D array or list of 
    2-D arrays. Means are calculated along the first (0) dimension. Normalize
    mean absolute value to mean_abs_val along each dimension.
    
    Args:
        x: 2-D array or list of 2-D arrays.
        
        mean_abs_norm: Positive value to normalize mean absolute value to. 
        Positive float. Can also be list/array of floats if each dimension is
        to have a different normalization constant.
        
    Returns:
        y: Detrended array/list of arrays.
        
    Example:
        >>> x = np.array([[2.,4.],[0.,8.],[4.,0.]])
        >>> detrend(x)
        array([[ 0. ,  0. ],
               [-1.5,  1.5],
               [ 1.5, -1.5]])
        >>> y = np.array([[4.,0.],[0.,8.],[2.,4.]])
        >>> detrend([x,y])
        
    """
    
    # Is x a list?
    if not isinstance(x,list):
        # Is mean_abs_norm a list?
        if isinstance(mean_abs_norm, int) or isinstance(mean_abs_norm, float):
            mean_abs_norm = [mean_abs_norm for ii in range(x.shape[1])]
        # Make sure x is a float
        x = x.astype(float)
        # Calculate mean
        mean_x = np.mean(x,0)
        # Subtract mean from x
        x -= mean_x
        # Calculate mean absolute value
        mean_abs_x = np.mean(np.abs(x),0)
        # Normalize each column of x
        for jj in range(x.shape[1]):
            x[:,jj] /= mean_abs_x[jj]
            x[:,jj] *= mean_abs_norm[jj]
    
    else:
        # Is mean_abs_norm a list?
        if isinstance(mean_abs_norm, int) or isinstance(mean_abs_norm, float):
            mean_abs_norm = [mean_abs_norm for ii in range(x[0].shape[1])]
        # Make sure elements of x are all float
        x = [x[ii].astype(float) for ii in range(len(x))]
        # Calculate mean
        mean_x = np.mean(np.concatenate(x),0)
        # Subtract mean from x
        x = [x[ii] - mean_x for ii in range(len(x))]
        # Calculate mean absolute value
        mean_abs_x = np.mean(np.abs(np.concatenate(x)),0)
        # Normalize each column of x
        for jj in range(x[0].shape[1]):
            for ii in range(len(x)):
                x[ii][:,jj] /= mean_abs_x[jj]
                x[ii][:,jj] *= mean_abs_norm[jj]
                
    return x

def pos_inv(x):
    """Calculate 1/x for positive x, returning np.inf for elements of x equal 
    to zero.
    
    Args:
        x: Array of non-negative floats.
    
    Returns:
        1/x, with 1/0 replaced by np.inf
    """
    # Set to floats
    x = x.astype(float)
    
    # Make return array of same shape as x
    y = np.empty(x.shape,dtype=float)
    
    # Set x's zero elements to np.inf
    y[x == 0] = np.inf
    # Set other elements to 1/x
    y[x != 0] = 1./x[x != 0]
    
    return y
    
def entropy(P):
    """Calculate the entropy of a probability distribution.
    
    If the probability distribution is not normalized, this function will 
    assume that it is actually a probability density.
    
    Args:
        P: Array of probabilities. Arbitrary dimensionality accepted.
    Returns:
        Entropy of distribution.
        
    Example:
        >>> P1 = np.array([.25,.25,.25,.25])
        >>> P2 = np.array([.1,.1,.4,.4])
        >>> P3 = .25*np.ones((100,))
        >>> entropy(P1)
        1.3862943611198906
        >>> entropy(P2)
        1.1935496040981333
        >>> entropy(P3)
        1.3862943611198919
    """
    
    # Reshape distribution into 1D array
    P = P.reshape((P.size,)).astype(float)
    P_sum = np.sum(P)
    # Calculate differential width
    dx = 1./P_sum
    # Calculate probability times log probability
    P_log_P = np.zeros(P.shape)
    P_log_P[P != 0] = P[P != 0]*np.log(P[P != 0])
    # Set nans/infs to 0
    P_log_P[np.isnan(P_log_P) + np.isinf(P_log_P)] = 0.
    # Calculate entropy
    ent = -np.sum(P_log_P)*dx
    
    return ent
    
def var_discrete(P):
    """Calculate the variance of a discrete probability distribution.
    
    The variance of a multidimensional probability distribution is the trace of
    the covariance matrix. All intervals between points are assumed to be unity
    i.e., dx = dy = ... = 1
    
    Args:
        P: Array of probabilities. Arbitrary dimensionality accepted.
    Returns:
        Variance of distribution.
    
    Example:
        >>> P1 = np.array([.2,.2,.2,.2,.2])
        >>> var_discrete(P1)
        2.0
    """
    
    # Make sure probability distribution is normalized
    P_norm = P.astype(float) / np.sum(P)
    # Create meshgrids
    if len(P_norm.shape) > 1:
        M = np.meshgrid(*[np.arange(0,n) for n in P_norm.shape])
    else:
        M = [np.arange(0,P_norm.shape[0])]
    # Calculate variance along each dimension and sum
    V = 0.
    for d in range(len(P_norm.shape)):
        # Get mean
        mu = np.sum(P_norm*M[d])
        # Get variance
        V += np.sum(P_norm*(M[d] - mu)**2)
    
    return V

def prob_from_log_like(log_like):
    """ Calculate probability from log_likelihoods, assuming flat prior.

    This is useful when log_likelihoods are not well-behaved, as it uses an
    efficient algorithm for calculating the logarithm of a sum from the 
    logarithms of its terms without raising numerical errors.
    
    Args:
        log_like: Array of log-likelihoods for all source positions.
        
    Returns:
        Probability of source at all possible positions.
    """
    # Calculate log[normalization factor] (log of summed likelihood)
    log_norm_factor = logsum(log_like.reshape(-1))
    # Calculate log of source probability
    log_prob = log_like - log_norm_factor
    # Calculate source probability
    prob = np.exp(log_prob)
    
    return prob

def nans(shape):
    """Create an array of nans.
    
    Useful for allocating space for a matrix.
    
    Args:
        shape: Shape of array to create. Tuple.
        
    Returns:
        x: nan array
        
    Example:
        >>> x = nans((3,3))
        >>> x
        array([[ nan,  nan,  nan],
               [ nan,  nan,  nan],
               [ nan,  nan,  nan]])
    """
    x = np.empty(shape,dtype=float)
    x[:] = np.nan
    return x
    
def nan_extend(mat,axis=0):
    """Double the size of a floating-point matrix using nans.
    
    Original elements are left unchanged. Nans are added. 
    
    Args:
        mat: Matrix to double size of.
        
        axis: Which axis to expand matrix along.
        
    Returns:
        mat: Matrix expanded to have double the size.
        
    """
    mat = np.concatenate([mat,nans(mat.shape)],axis)
    return mat
    
def symtri(x,center=0,height=1,slope=1):
    """Symmetric triangle function.
    
    Returns the value of a symmetric trianglel function evaluated at x. This is
    zero if x is less than the intersection of the left arm of the triangle
    with the x-axis or if x is greater than the intersection fo the right arm
    of the triangle with the x-axis. Otherwise it returns the linear function
    of x corresponding to the arm in which x is located.
    
    """
    # Calculate x-intercepts
    x_left = center - height/slope
    x_right = center + height/slope
    
    if x > x_right or x < x_left:
        return 0
    elif x >= x_left and x < center:
        return (x - x_left)*slope
    elif x <= x_right and x >= center:
        return (x_right - x)*slope
        
def cartesian_product(x,*args):
    """Return Cartesian product of two 1D arrays.
    
    Args:
        x: One array.
        
        args: Other arrays. If none provided, cartesian product of x and x
        will be returned.
        
    Returns:
        Cartesian product of x and y.
        
    Example:
        >>> x = np.array([1,3,5])
        >>> y = np.array([2,4,6])
        >>> cartesian_product(x,y)
        array([[1, 2],
               [3, 2],
               [5, 2],
               [1, 4],
               [3, 4],
               [5, 4],
               [1, 6],
               [3, 6],
               [5, 6]])
    """
    if args:
        args = [x] + list(args)
    else:
        args = [x,x]
        
    A = np.meshgrid(*args)
    A = [A[ii].reshape(-1,1) for ii in range(len(A))]
    return np.concatenate(A,1)
    
def mat_prod(mat_list):
    """Calculate the product of several matrices.
    
    Matrices are given as numpy arrays. One dimensional arrays are treated as
    column vectors.
    
    Args:
        mat_list: List of numpy arrays, in the order that they are to be
        multiplied.
        
    Returns:
        Matrix product of all matrices in mat_list.
        
    Example:
        >>> X1 = np.array([[2,3],[1,2]])
        >>> X2 = np.array([[6,1],[-3,2]])
        >>> X3 = np.array([[-5,-5],[2,6]],dtype=float)
        >>> mat_prod([X1,X2,X3])
        array([[  1.,  33.],
               [ 10.,  30.]])
        >>> np.dot(X1,np.dot(X2,X3))
        array([[  1.,  33.],
               [ 10.,  30.]])
    """

    # Convert all 1-D arrays to column vector
    for mat_idx,mat in enumerate(mat_list):
        if len(mat.shape) == 1:
            mat_list[mat_idx] = mat.reshape((-1,1))
            
    Y = np.eye(mat_list[-1].shape[1])
    for mat in mat_list[::-1]:
        Y = np.dot(mat,Y)
    return Y

def psd(t_series,n=None):
    """Approximate the power spectral density (PSD) of a list of time-series.
    
    The PSD is approximated by calculating the power spectrum of each time-
    series and averaging them together.
    
    Args:
        t_series: List of time-series. If vector time-series, each must be a 
        numpy array with rows indicating times and columns indicating 
        variables.
        
        n: Number of points in psd. Standard conventions of zero-padding and
        truncation are used.
        
    Returns:
        Power spectral density for each dimension in each time-series
        
    Example:
        >>> t = np.arange(0,10,.1)
        >>> t_series = [(3+np.random.normal(0,1,1))*sin(3*t) + \
                        (5+np.random.normal(0,1,1))*sin(5*t) \
                        for ii in range(100)]
        >>> t_series = [np.tile(X,(2,1)).T for X in t_series]
        >>> t_series = [X + np.random.normal(0,.2,X.shape) for X in t_series]
        >>> plt.figure()
        >>> for ii in range(100):
        ...     plt.plot(t_series[ii])
        >>> p_spec_dens = psd(t_series)
        >>> plt.figure()
        >>> plt.plot(p_spec_dens)
    """
    
    if n is None:
        n = max([X.shape[0] for X in t_series])
        
    # Convert all time-series to two-dim arrays
    if len(t_series[0].shape) == 1:
        t_series = [np.array([X]).T for X in t_series]
        
    # Calculate DFT of each time-series
    dft = [np.fft.fft(X,n=n,axis=0) for X in t_series]
    
    # Calculate power spectrum
    ps = [FT*np.conj(FT) for FT in dft]
    
    # Calculate psd by taking mean
    return np.mean(np.array(ps),axis=0)
    
def func_dist_uneven(x1,y1,x2,y2,p=2.,d=10):
    """Return the LP distance between two functions that are defined over 
    different supports.
    
    x1 and x2 must be strictly increasing. All function values are assumed to
    be zero outside of the region of their provided support.
    
    Args:
        x1: Support of first function.
        
        y1: Values of first function.
        
        x2: Support of second function.
        
        y2: Values of second function.
        
        p: Which LP norm to use.
        
        d: Resolution parameter. The dx used in calculating the distance will be
        the minimum dx in either x1 or x2 divided by d.
        
    Returns:
        Distance between two functions.
        
    Example:
        >>> x1 = np.linspace(0,10,30)
        >>> x2 = np.linspace(0,10,100)
        >>> y1 = 4*np.sin(x1)
        >>> y2 = 2*np.cos(x2)
        >>> func_dist_uneven(x1,y1,x2,y2)
        9.441791672756592
    """
    
    # Get dx (minimum x window divided by d)
    dx = np.min(np.concatenate([np.diff(x1),np.diff(x2)])) / d
    # Get xmin and xmax
    xmin = np.min(np.concatenate([x1,x2]))
    xmax = np.max(np.concatenate([x1,x2]))
    
    x1c = x1.copy()
    x2c = x2.copy()
    y1c = y1.copy()
    y2c = y2.copy()
    
    if xmin < np.min(x1c):
        x1c = np.concatenate([np.array([xmin]),x1c])
        y1c = np.concatenate([np.array([0.]),y1c])
    if xmin < np.min(x2c):
        x2c = np.concatenate([np.array([xmin]),x2c])
        y2c = np.concatenate([np.array([0.]),y2c])
    if xmax > np.max(x1c):
        x1c = np.concatenate([x1c,np.array([xmax])])
        y1c = np.concatenate([y1c,np.array([0.])])
    if xmax > np.max(x2c):
        x2c = np.concatenate([x2c,np.array([xmax])])
        y2c = np.concatenate([y2c,np.array([0.])])

    # Generate new support
    x = np.arange(xmin,xmax,dx)
    
    # Build interpolated functions
    f1 = interpolate.interp1d(x1c,y1c)
    f2 = interpolate.interp1d(x2c,y2c)
    
    # Calculate new y values of each function using interpolater
    y1_int = f1(x)
    y2_int = f2(x)
        
    # Calculate distance
    return (np.sum((y1_int - y2_int)**p)*dx)**(1./p)
    
def cdiff(a, axis=0):
    """Calculate the central differences of vectors in a along a given axis.
    
    Args:
        a: Array.
        axis: Which axis to calculate central differences along.
    Returns:
        Array of central differences, taken along specified axis.
    Example:
        >>> x = np.arange(40).reshape(10,4)
        >>> cdiff(x,0)
        array([[ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.],
               [ 4.,  4.,  4.,  4.]])
        >>> cdiff(x,1)
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]])        
        
    """
    
    diff_array = np.diff(a,n=1,axis=axis).astype(float)
    append_shape = list(a.shape)
    append_shape[axis] = 1
    append_array = nans(append_shape)
    
    # Take average of forwards and backwards differences
    f_diff = np.concatenate([diff_array,append_array],axis)
    b_diff = np.concatenate([append_array,diff_array],axis)
    f_nan_mask = np.isnan(f_diff)
    b_nan_mask = np.isnan(b_diff)
    f_diff[f_nan_mask] = b_diff[f_nan_mask]
    b_diff[b_nan_mask] = f_diff[b_nan_mask]
    
    return (f_diff + b_diff)/2.
    
def edges(x):
    """Calculate edges of groups of nonzero elements in a 1D array.
    
    Args:
        x: Array.
    Returns:
        2D array of edges. First column is lower edge, second column is upper
        edge.
    Example:
        >>> x = np.array([0,0,0,1,1.5,2,1.5,1,0,0,0,-2,-2.5,-2.3,0,0])
        array([[3,8],
               [11,14]])
    """
    
    new_x = np.concatenate([np.array([0]),(x != 0).astype(int),np.array([0])])
    d = np.diff(new_x)
    lower = (d == 1).ravel().nonzero()[0]
    upper = (d == -1).ravel().nonzero()[0]
    return np.array([lower,upper]).T
    
def dkl(P1,P2,dx=1.,sym=False):
    """Calculate the Kullback-Leibler divergence (DKL) between two 
    distributions with the same domain.
    
    Args:
        P1: First distribution.
        P2: Second distribution.
        dx: Integration interval for continuous distributions.
        sym: Set to True to get symmetric DKL
        
    Returns:
        DKL between the two distributions.
    """
    
    dkl1 = np.sum(np.log(P1/P2)*P1)*dx
    
    if sym:
        dkl2 = np.sum(np.log(P2/P1)*P2)*dx
        return (dkl1 + dkl2)/2.
    else:
        return dkl1
        
def ks_stat(P1,P2,dx=1.):
    """Calculate the Kolmogorov-Smirnov statistic for two distributions.
    
    Args:
        P1: First distribution.
        P2: Second distribution.
        dx: Integration interval for continuous distributions.
    
    Returns:
        KS statistic for P1 and P2.
    """
    
    # Calculate cumulative distributions
    C1 = np.cumsum(P1)*dx
    C2 = np.cumsum(P2)*dx
    
    # Find maximum distance between cumulative distributions
    return np.max(np.abs(C2-C1))
    
def peaks(x):
    """Find the peaks of the vector array x.
    
    Args:
        x: 1D array.
    
    Returns:
        Logical indices of peaks, values of peaks.
        
    Example:
        >>> x = np.array([0,2,3,4,2,2,-1,-2,-1,2,5,8,4,3,5,1])
        >>> idx, pks = peaks(x)
        >>> idx
        array([False, False, False,  True, False, False, False, False, False,
               False, False,  True, False, False,  True, False], dtype=bool)
        >>> pks
        array([4, 8, 5])
    """
    
    # Calculate derivative
    dx = np.diff(x)
    # Binarize derivative
    dx[dx > 0] = 1.
    dx[dx <= 0] = 0.
    # Get second derivative
    d2x = np.concatenate([np.array([0.]),np.diff(dx),np.array([0.])])
    # Find indices of peaks
    idx = (d2x == -1.)
    pks = x[idx]
    
    return idx,pks
    
def nanmax(x,*args,**kwargs):
    """Return the maximum of an array or nan if the array is empty.
    
    Args:
        x: Array (possibly empty)
        
    Returns:
        Max of array or nan if array is empty.
        
    Example:
        >>> nanmax(np.array([1,3,6,1]))
        6
        >>> nanmax(np.array([[1,3,6,1],[1,7,3,4]]),1)
        array([6, 7])
        >>> nanmax(np.array([]))
        nan
    """
    
    if x.size:
        return np.max(x,*args,**kwargs)
    else:
        return np.nan

def nanmin(x,*args,**kwargs):
    """Return the minimum of an array or nan if the array is empty.
    
    Args:
        x: Array (possibly empty)
        
    Returns:
        Max of array or nan if array is empty.
        
    Example:
        >>> nanmin(np.array([1,3,6,1]))
        1
        >>> nanmin(np.array([[1,3,6,1],[1,7,3,4]]),1)
        array([1,1])
        >>> nanmin(np.array([]))
        nan
    """
    
    if x.size:
        return np.min(x,*args,**kwargs)
    else:
        return np.nan
    
def fit_nonlinearity(x,y,approx='piecewise_linear',bins=10,equal_inputs=True):
    """Fit an arbitrary nonlinearity to a dataset.
    
    Fit dataset with a nonlinear curve.
    
    Args:
        x: x-coordinates of data
        y: y-coordinates of data
        approx: which approximation to use to define nonlinearity. Options: 
            'piecewise_linear',...
        bins: how many bins to use in approximating the nonlinearity
        equal_inputs: Whether or not to use an equal number of inputs in each 
        bin.
    Returns:
        continuous function defined over domain of x.
    Example:
        >>> x = np.arange(0,5,.01)
        >>> y = np.exp(x) + np.random.normal(0,2,x.shape)
        >>> plt.scatter(x,y)
        >>> f = fit_nonlinearity(x,y,bins=10)
        >>> plt.plot(x,f(x),'r',linewidth=2)
    """
    
    # Sort x and y according to x
    sort_idx = x.argsort()
    xs = x[sort_idx]
    ys = y[sort_idx]
    
    per_bin = np.ceil(len(xs)/bins)
    bin_means = nans((bins,))
    bin_cents = nans((bins,))
    slopes = nans((bins,))
    intercepts = nans((bins,))
    for bin_idx in range(bins):
        bin_start = bin_idx*per_bin
        bin_end = (bin_idx+1)*per_bin
        x_bin = xs[bin_start:bin_end]
        y_bin = ys[bin_start:bin_end]
        bin_cents[bin_idx] = x_bin.mean()
        bin_means[bin_idx] = y_bin.mean()
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_bin,y_bin)
        slopes[bin_idx] = slope
        intercepts[bin_idx] = intercept

    bin_cents[0] = min(x)
    bin_cents[-1] = max(x)
    # Create nonlinear function        
    def f(z_array):
        out_array = nans(z_array.shape)
        for z_idx,z in enumerate(z_array):
            for bin_idx in range(bins-1):
                if bin_cents[bin_idx] <= z <= bin_cents[bin_idx+1]:
#                    dy = bin_means[bin_idx+1] - bin_means[bin_idx]
#                    dx = bin_cents[bin_idx+1] - bin_cents[bin_idx]
#                    slope = dy/dx
#                    intercept = bin_means[bin_idx] - slope*bin_cents[bin_idx]
                
                    slope = slopes[bin_idx]
                    intercept = intercepts[bin_idx]
                    out_array[z_idx] = slope*z + intercept
        return out_array
    
    return f