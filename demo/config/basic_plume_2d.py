import numpy as np
from plume import Environment3d
from logprob_odor import binary_advec_diff_tavg


# PLOTTING
PLOTEVERY = 10
PAUSEEVERY = 50
PLOTKWARGS = {'figsize': (10, 10),
              'facecolor': 'w'}


# SIMULATION
RUNTIME = 100  # (s)


# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 2)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)


# PLUME
SRCPOS = (0, 7, 0)
W = 0.4  # wind (m/s)
R = 10  # source emission rate
D = 0.12  # diffusivity (m^2/s)
A = .002  # searcher size (m)
TAU = 1000  # particle lifetime (s)

# DEBUGGING PARAMS
# W = 0.4
# R = 100
# D = .001
# A = .00001
# TAU = 25


# INSECT
STARTPOS = (64, 7, 0)
LOGLIKE = binary_advec_diff_tavg