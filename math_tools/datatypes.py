import pdb
"""
Modified   Apr 17 2014
Created on Apr 11 2014

@author: rkp

This encodes several useful mathematical data types.
"""

import numpy as np

class Space():
    """Space data type.
    
    Contains meshgrid-style information about space.
    
    Args:
        edges: Specification of edges in each dimension. Syntax - 
            np.array([xmin,xmax,ymin,ymax,zmin,zmax]).
    """
    
    def __init__(self,edges=np.array([0,512,0,512]),step=1):
        """Space constructor."""
        
        # Get dimensionality
        self.dim = len(edges)/2
        # Create meshgrids
        self.X_idx = None
        self.Y_idx = None
        self.Z_idx = None
        if self.dim == 2:
            self.X, self.Y = np.mgrid[edges[0]:edges[1]:step,
                                      edges[2]:edges[3]:step]
            self.sparse = [np.arange(edges[0],edges[1],step),
                           np.arange(edges[2],edges[3],step)]
        elif self.dim == 3:
            self.X, self.Y, self.Z = np.mgrid[edges[0]:edges[1],
                edges[2]:edges[3],edges[4]:edges[5]]
            self.sparse = [np.arange(edges[0],edges[1],step),
                           np.arange(edges[2],edges[3],step),
                           np.arange(edges[4],edges[5],step)]
        # Store format as mgrid
        self.format = 'mgrid'
        # Store shape
        self.shape = self.X.shape
        # Calculate volume element
        self.dV = np.prod([self.sparse[ii][1] - self.sparse[ii][0] \
            for ii in range(self.dim)])
            
    def near(self,pos,idx_set=False):
        """Find nearest position in space to pos.
        
        Args:
            pos: Position vector to approximate.
            
            idx_set: Whether or not to return just index.
            
        Returns:
            Position rounded to nearest point in space (tuple).
        """
        
        # Get index nearest to pos
        idx = tuple([np.argmin(np.abs(pos[ii] - self.sparse[ii])) \
            for ii in range(self.dim)])
        
        # Just return index, or position associated with index?
        if idx_set:
            return idx
        else:
            return tuple([self.sparse[ii][idx[ii]] for ii in range(self.dim)])
    
    def in_rad(self,pos,rad,idx_set=False):
        """Get a list of positions/indices within radius of pos.
        
        Args:
            pos: Position vector.
            
            rad: Radius.
            
            idx_set: Whether or not to return just indices (instead of pos's).
            
        Returns:
            Array of positions/indices within rad of pos. Each row is a 
            position/index.
            
        Example:
            >>> S = Space(np.array([-5,5,-5,5.]),step=.5)
            >>> pos = np.array([1.,1.])
            >>> S.in_rad(pos,.6)
            array([[ 0.5,  1. ],
                   [ 1. ,  0.5],
                   [ 1. ,  1. ],
                   [ 1. ,  1.5],
                   [ 1.5,  1. ]])
            >>> S.in_rad(pos,.6,idx_set=True)
            array([[11, 12],
                   [12, 11],
                   [12, 12],
                   [12, 13],
                   [13, 12]])
            >>> S.in_rad(pos,.1)
            array([[ 1.,  1.]])
        """
        
        dX = (self.X - pos[0]).astype(float)
        dY = (self.Y - pos[1]).astype(float)
        
        if self.dim == 2:
            mask = (dX**2 + dY**2) <= rad**2
        elif self.dim == 3:
            dZ = self.Z - pos[2]
            mask = dX**2 + dY**2 + dZ**2 <= rad**2
        
        if idx_set:
            # Generate meshgrids of indices if not set
            self.gen_idx_mgrid()
            # Get indices for all positions indicated by mask
            x = self.X_idx[mask]
            y = self.Y_idx[mask]
            if self.dim == 3:
                z = self.Z_idx[mask]
        else:
            # Get indices for all positions indicated by mask
            x = self.X[mask]
            y = self.Y[mask]
            if self.dim == 3:
                z = self.Z[mask]

        # Get lists of indices/positions marked by mask
        if self.dim == 2:
            if np.sum(mask) == 1:
                return np.array([x,y]).T
            else:
                return np.concatenate([x,y]).reshape((2,-1)).T
        elif self.dim == 3:
            if np.sum(mask) == 1:
                return np.array([x,y,z]).T
            else:
                return np.concatenate([x,y,z]).reshape((3,-1)).T
    
    def get_bdry(self):
        """Return a dictionary of boundaries.
        
        Returns:
            2D array of boundaries. Rows are dimensions. First column is lower
            boundary, second column is upper boundary.
        """
        
        if self.dim == 2:
            bdry = np.array([[self.sparse[0][0],self.sparse[0][-1]],
                             [self.sparse[1][0],self.sparse[1][-1]]])
        elif self.dim == 3:
            bdry = np.array([[self.sparse[0][0],self.sparse[0][-1]],
                             [self.sparse[1][0],self.sparse[1][-1]],
                             [self.sparse[2][0],self.sparse[2][-1]]]) 
        return bdry
        
    def gen_idx_mgrid(self):
        """Generate meshgrids of indices."""
        
        if self.X_idx is None:
            X_bnd = [0,self.shape[0]]
            Y_bnd = [0,self.shape[1]]
            if self.dim == 2:
                self.X_idx,self.Y_idx = np.mgrid[X_bnd[0]:X_bnd[1],
                                                 Y_bnd[0]:Y_bnd[1]]
            elif self.dim == 3:
                Z_bnd = [0,self.shape[2]]
                self.X_idx,self.Y_idx,self.Z_idx = \
                    np.mgrid[X_bnd[0]:X_bnd[1],
                             Y_bnd[0]:Y_bnd[1],
                             Z_bnd[0]:Z_bnd[1]]
        
           
class ParArray():
    """Parametric array data type.
    
    The parametric array allows one to store data in an array using either the 
    numpy array structure itself or a parametric representation. In the 
    latter case, a function must also be set that indicates how the parameters
    should be used to generate points in the array.
    
    Args:
        Y: Either numpy array or dictionary of parameters.
        
        D: Dimensionality of array.
        
        X: Support (necessary if Y is an array). List of 1-D numpy arrays.
        
        F: Function converting dictionary of parameters to numpy array over
        specified support.
        
    Example:
        >>> Y = {'a':3, 'b':5, 'c':-1}
        >>> def F(Y,x1,x2): Y['a']*x1 + Y['b']*x2 + Y['c']
        >>> q = ParArray(Y=Y,D=1,F=F)
        
    """
    
    def __init__(self,Y,D=2,X=None,F='array'):
        """ParArray constructor."""
        
        self.Y = Y
        self.F = F
        # If Y is not a parametric rep...
        if not isinstance(Y,dict):
            D = len(Y.shape)
            # Has support not been provided?
            if X is None:
                X = [np.arange(N) for N in Y.shape]
            self.X = X
            self.dV = np.prod([self.X[ii][1] - self.X[ii][0] \
                for ii in range(D)])
            self.size = Y.size
            self.shape = Y.shape
        self.D = D
        
    def arr(self,X=None):
        """Return an array.
        
        Args:
            X: Support of array. List of 1-D numpy arrays (one per dimension).
            
        Returns:
            Array calculated over support, or stored array.
            
        """
        if X is None:
            return self.Y
        else:
            # Create meshgrids
            X_mesh = np.meshgrid(X)
            # Calculate array over support defined by X_mesh
            return self.F(self.Y,*X_mesh)
            
    def is_par(self):
        """Say whether parametric or not.
        
        Returns:
            True if parametric, False otherwise.
            
        """
        return self.F != 'array'
            
    def plus(self,q):
        """Add another parametric array to this one.
        
        Args:
            q: Parametric array instance. Can also be numpy array object.
            
        """
        if hasattr(q,'D'): # ParArray
            if not (q.is_par() or self.is_par()):
                self.Y += q.arr
            else:
                pass
        else: # Numpy array
            self.Y += q
        
    def minus(self,q):
        """Subtract another parametric array from this one.
        
        Args:
            q: Parametric array instance.
            
        """
        if not (q.is_par() or self.is_par()):
            self.Y -= q.arr()
        else:
            pass
        
    def set_val(self,pos,val,idx_set=False):
        """Set an element of a non-parametric array to val.
        
        Args:
            pos: Tuple containing position.
            
            val: Value to set it to.
            
            idx_set: Whether or not to interpet pos as index.
            
        """
        # Is pos an index?
        if idx_set:
            idx = pos
        else:
            # Find index closest to specified position
            idx = np.array([np.argmin(np.abs(pos[ii] - self.X[ii])) \
                for ii in range(self.D)]).astype(int)
            
        self.Y[tuple(idx)] = val