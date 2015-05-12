import numpy as np
from plume import Environment3d
from logprob_odor import binary_advec_diff_tavg

ID = 'wind_tunnel_basic_plume_R{}'
DESCRIPTION = ('Trials generated in a wind tunnel with a plume based on the'
               ' advection-diffusion equation, with an R specified in the'
               ' simulation id. Insect\'s model of plume is perfectly matched'
               ' to true plume used in simulation.')
GEOMCONFIGGROUP = 'wind_tunnel_random_src_and_start'

HEADING_SMOOTHING = 2

# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)

# PLUME
PLUME_PARAMS = {
                'w': 0.4,  # wind (m/s)
                'r': None,  # source emission rate
                'd': 0.12,  # diffusivity (m^2/s)
                'a': .002,  # searcher size (m)
                'tau': 1000,  # particle lifetime (s)
                }

Rs = [5, 10, 50, 100, 500, 1000, 5000]

# INSECT
LOGLIKE = binary_advec_diff_tavg