import numpy as np
from plume import Environment3d
from logprob_odor import binary_advec_diff_tavg

ID = 'test_simulation'
DESCRIPTION = 'Test simulation. Generate and save 5 infotaxis runs in the database.'
GEOMCONFIGGROUP = 'test_geom_config_group'

# PLOTTING
PLOTEVERY = 10
PAUSEEVERY = 0
PLOTKWARGS = {'figsize': (10, 10),
              'facecolor': 'w'}


# SIMULATION
TOTALTRIALS = 5


# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)


# PLUME
SRCPOS = (0, 7, 7)
PLUME_PARAMS = {
                'w': 0.4,  # wind (m/s)
                'r': 500,  # source emission rate
                'd': 0.12,  # diffusivity (m^2/s)
                'a': .002,  # searcher size (m)
                'tau': 1000,  # particle lifetime (s)
                }


# INSECT
STARTPOS = (64, 0, 0)
LOGLIKE = binary_advec_diff_tavg