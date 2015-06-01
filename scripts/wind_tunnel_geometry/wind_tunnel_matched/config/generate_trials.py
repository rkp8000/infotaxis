import numpy as np

from plume import Environment3d
from logprob_odor import binary_advec_diff_tavg

SIMULATIONID = 'wind_tunnel_matched_fruitfly_0.4mps_odor_on'
SIMULATIONDESCRIPTION = 'Creates large set of trajectories with starting positions and durations drawn randomly from ' \
                        'true wind tunnel dataset for fruit flies flying in 0.4 m/s wind with odor present.'

GEOMCONFIGGROUPID = 'wind_tunnel_matched_fruitfly_0.4mps_checkerboard_floor_odor_on'
# make sure to set max plume conc to zero if no odor plume!

HEADINGSMOOTHING = 3

TOTALTRIALS = 1000

# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)

# PLUME PARAMS
PLUMEPARAMS = {
               'max_conc': 488,
               'threshold': 10,
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
