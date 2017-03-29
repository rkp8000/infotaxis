"""
Created on Thu Aug  7 12:58:09 2014

@author: rkp

Functions for managing arrays
"""

import numpy as np

def wb_row_idx(x):
    """Get indices of all rows in a 2D array that contain well-behaved entries
    (contain neither a nan nor an inf).
    
    Args:
        x: 2D array with potential nan/inf values.
    Returns:
        Indices of all rows without nan or inf
    Example:
        >>> x = np.array([[1,3,6],[3,1.,2],[np.nan,3,1],[6,np.inf,3],[1,1,1]])
        >>> x[wb_row_idx(x),:]
        array([[ 1.,  3.,  6.],
               [ 3.,  1.,  2.],
               [ 1.,  1.,  1.]])
    """
    
    if len(x.shape) == 1:
        no_nan_idx = ~np.isnan(x)
        no_inf_idx = ~np.isinf(x)
    else:
        # Get logical indices of all not-nan rows
        no_nan_idx = np.all(~np.isnan(x),1)
        # Get logical indices of all not-inf rows
        no_inf_idx = np.all(~np.isinf(x),1)
    # Get logical AND operation between these two vectors
    return no_nan_idx * no_inf_idx
    
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
    
def nearest(x, a):
    """Return the index of the element of a nearest to x.
    
    Useful for identifying indices of specific elements when floating
    point errors are involved."""
    
    return np.argmin(np.abs(np.array(a) - x))