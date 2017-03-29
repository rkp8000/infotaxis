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

def load_mat_struct(filename):
    """ Load a .mat file containing a single structure and convert the
    structure to a python dict.
    
    Args:
        filename: Path of the .mat file to load.
    
    Returns:
        mat_dict: Dictionary containing same information as .mat struct.
        
    Example:
        >>> S = load_mat_struct('/Users/admin/Documents/rkp/testdata.mat')
        >>> S
    """
    # Load mat file
    mat = io.loadmat(filename)
    
    # Get structure name
    struct_name = mat.keys()[0]
    
    # How many fields in the structure (converted to dict keys)
    num_fields = len(mat[struct_name][0][0])
    
    # Instantiate new dictionary
    mat_dict = {}
    
    # Fill in keys and values
    for ii in range(num_fields):
        mat_dict[ii] = mat[struct_name][0][0][ii]
        
    return mat_dict