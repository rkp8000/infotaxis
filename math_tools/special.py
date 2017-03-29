"""
Created on Mon Jan 12 15:41:36 2015

@author: rkp

Special functions. Usually offshoots of special functions in scipy.
"""

import numpy as np
from scipy.special import k0, kn

def logk0(x):
    """Logarithm of modified bessel function of the second kind of order 0. 
    Infinite values may still be returned if argument is too close to 
    zero."""
    
    y = k0(x)
    
    # if array
    try:
        xs = x.shape
        logy = np.zeros(x.shape, dtype=float)
        
        # attempt to calculate bessel function for all elements in x
        logy[y!=0] = np.log(y[y!=0])
        
        # for zero-valued elements use approximation
        logy[y==0] = -x[y==0] - np.log(x[y==0]) + np.log(np.sqrt(np.pi/2))
        
        return logy
        
    except:
        if y == 0:
            return -x - np.log(x) + np.log(np.sqrt(np.pi/2))
        else:
            return np.log(y)
            
            
def logkn(n, x):
    """Logarithm of modified bessel function of the second kind of integer
    order n. Infinite values may still be returned if argument is too close to 
    zero."""
    
    y = kn(n, x)
    
    # if array
    try:
        xs = x.shape
        logy = np.zeros(x.shape, dtype=float)
        
        # attempt to calculate bessel function for all elements in x
        
        logy[y!=0] = np.log(y[y!=0])
        
        # for zero-valued elements use approximation
        logy[y==0] = -x[y==0] - np.log(x[y==0]) + np.log(np.sqrt(np.pi/2))
        
        return logy
        
    except:
        if y == 0:
            return -x - np.log(x) + np.log(np.sqrt(np.pi/2))
        else:
            return np.log(y)