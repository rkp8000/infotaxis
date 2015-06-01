import numpy as np

from plume import Environment3d
from logprob_odor import binary_advec_diff_tavg

# PLOTTING
PLOTEVERY = 10
PAUSEEVERY = 0
PLOTKWARGS = {'figsize': (10, 10),
              'facecolor': 'w'}

# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)

# PLUME PARAMS
PLUMEPARAMS = {
               'max_conc': 488,
               'threshold': 0.1,
               'ymean': 0.0105,
               'zmean': 0.0213,
               'ystd': 0.0073,
               'zstd': 0.0094
               }

# INSECT PARAMS
INSECTPARAMS = {
                'w': 0.4,  # wind (m/s)
                'r': 1000,  # source emission rate
                'd': 0.12,  # diffusivity (m^2/s)
                'a': .002,  # searcher size (m)
                'tau': 10000  # particle lifetime (s)
                }
LOGLIKE = binary_advec_diff_tavg

STARTPOSIDX = (np.random.randint(0, len(XRBINS) - 2),
               np.random.randint(0, len(YRBINS) - 2),
               np.random.randint(0, len(ZRBINS) - 2))

DURATION = 1000