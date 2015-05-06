import numpy as np
from plume import Environment3d
from logprob_odor import binary_advec_diff_tavg


# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 2)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)

# DEBUGGING PARAMS
W = 0.4
R = 100
D = .001
A = .00001
TAU = 25

LOGLIKE = binary_advec_diff_tavg