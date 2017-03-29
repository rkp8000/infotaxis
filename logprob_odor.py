"""
Created on Fri Dec  5 11:58:13 2014

@author: rkp

Contains functions used to calculate the probability of an odor encounter given
a certain type of turbulent statistics. All functions return the log probability
of an odor concentration for an array of different possible source positions.
"""

import numpy as np
from math_tools.special import logk0

def advec_diff_mean_hit_rate(dx, dy, dz, w, r, d, a, tau, dim=3):
    """Calculate the mean hit number at a displacement relative to the source.
    
    Args:
        dx: current x - source x
        dy: current y - source y
        dz: current z - source z
        w: wind speed (m/s, positive blows in +x direction)
        r: source emission rate(conc*m^2/s)
        d: diffusion coefficient (m^2/s)
        a: linear particle size (m)
        tau: particle lifetime
        dim: dimension of problem
    """
    
    # calculate absolute distance from source
    dr = (dx**2 + dy**2 + dz**2) ** .5
    
    # calculate lambda (correlation length)
    lam = np.sqrt((d*tau) / (1 + ((w**2) * tau) / (4*d)))
    
    # calculate hit rate & probability from source at all positions
    if dim == 2:  # 2D
        # raise error if particle size is larger than lambda
        if a > lam:
            raise ValueError('Particle size a (%.8f) cannot be larger than correlation length (%.8f)!' % (a, lam))

        exponent = w*dx / (2*d)
        lograte = np.log(r/np.log(lam/a)) + exponent + logk0(dr/lam)
        rate = np.exp(lograte)
        # replace nan result from bessel function with np.inf
        if type(rate) is np.ndarray:
            rate[np.isnan(rate)] = np.inf
    elif dim == 3:  # 3D
        exponent = (dx*w / (2*d)) - (dr/lam)
        rate = (a*r/dr) * np.exp(exponent)
    
    return rate
    
def binary_advec_diff_tavg(odor, pos_idx, xext, yext, zext, dt, w, r, d, a, tau):
    """Calculate the probability of measuring an odor value for a 3D array of
    possible source positions. Specifically, calculates probability of binary
    odor signal using time-averaged advection-diffusion equation. Assumes that
    given a hit rate, hit number is Poisson distributed.
    
    Args:
        odor: binary odor value (hit = 1, miss = 0)
        pos_idx: position array of integers, corresponding to xext, yext, zext
        xext: range of x-positions to try source in
        yext: range of y-positions to try source in
        zext: range of z-positions to try source in
        
        dt: time interval over which odor was averaged (s)
        w: wind speed (m/s, blowing in +x direction)
        r: source emission rate (conc*m^2/s)
        d: diffusion coefficent (m^2/s)
        a: linear particle size (m)
        tau: particle lifetime (s)
        
    Returns:
        3D array of probabilities of odor encounter for different source
        locations, with dimensions corresponding to xext, yext, and zext."""
    
    # calculate distance to all possible sources & make meshgrid arrays
    dx = xext - xext[pos_idx[0]]
    dy = yext - yext[pos_idx[1]]
    dz = zext - zext[pos_idx[2]]
    DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing='ij')
    
    if len(zext) == 1:
        dim = 2
    else:
        dim = 3
        
    mean_hit_num = dt * advec_diff_mean_hit_rate(-DX, -DY, -DZ, w, r, d, a, tau, dim=dim)
    
    Lmiss = -mean_hit_num
    Lhit = np.log(np.maximum(0., 1 - np.exp(-mean_hit_num)))
    
    # Calculate probability of odor at pidx for al possible source positions
    if odor:
        LPodor = Lhit
    else:
        LPodor = Lmiss      
    
    return LPodor
    
binary_advec_diff_tavg.domain = np.array([0, 1])