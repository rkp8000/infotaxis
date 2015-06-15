import numpy as np

from plume import Environment3d
from logprob_odor import binary_advec_diff_tavg

EXPERIMENTS = ('fruitfly_0.3mps_checkerboard_floor',
               'fruitfly_0.4mps_checkerboard_floor',
               'fruitfly_0.6mps_checkerboard_floor')
ODOR_STATES = ('on', 'none', 'afterodor')

SIMULATION_ID = 'wind_tunnel_discretized_matched_r{}_d{}_{}_odor_{}'
SIMULATION_DESCRIPTION = 'Set of trajectories with starting positions and durations exactly matched to wind tunnel data, but generated using the infotaxis algorithm. When plume is present, collimated plume is used that gives hits whenever concentration is greater than a certain threshold. For experiment "{}" with odor state "{}".'

WIND_TUNNEL_DISCRETIZED_SIMULATION_ID_PATTERN = '%wind_tunnel_discretized_copies%'

GEOM_CONFIG_GROUP_ID = 'wind_tunnel_matched_discretized_{}_odor_{}'
# make sure to set max plume conc to zero if no odor plume!

HEADING_SMOOTHING = 0

# ENVIRONMENT
DT = .06  # (s)
X_BINS = np.linspace(-.3, 1.0, 66)  # (m)
Y_BINS = np.linspace(-.15, .15, 16)  # (m)
Z_BINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(X_BINS, Y_BINS, Z_BINS)

# INSECT PARAMS
INSECT_PARAMS = {
                'r': 1000,  # source emission rate
                'd': 0.12,  # diffusivity (m^2/s)
                'a': .002,  # searcher size (m)
                'tau': 10000  # particle lifetime (s)
                }
LOGLIKE = binary_advec_diff_tavg
