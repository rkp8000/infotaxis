import os
import numpy as np

from plume import Environment3d
from logprob_odor import binary_advec_diff_tavg

# LOCAL PATHS
WT_REPO = os.path.join(os.getenv('REPOSITORY_DIRECTORY'), 'wind_tunnel')

# THINGS TO LOOP OVER
EXPERIMENT_IDS = ('fruitfly_0.3mps_checkerboard_floor',
                  'fruitfly_0.4mps_checkerboard_floor',
                  'fruitfly_0.6mps_checkerboard_floor',)
ODOR_STATES = ('on', 'none', 'afterodor')

# GEOM CONFIG GROUP
GEOM_CONFIG_GROUP_ID = 'wind_tunnel_matched_discretized'

# SIMULATION
SIMULATION_ID = 'wind_tunnel_discretized_copies_{}_odor_{}'
SIMULATION_DESCRIPTION = 'Load all wind tunnel trajectories and make discretized copies of them for direct comparison with infotaxis-generated trajectories. In addition to simply calculating the discretized sequence of position idxs, we assume that despite being forced to move along a certain trajectory, the insect is updating a belief distribution over source positions using what it knows about turbulent statistics. Specifically, it assumes a basic advection-diffusion time-averaged plume, as in Vergassolla 2007, and it calculates expected hit rate using a dt determined by dividing the total trajectory time by the number of steps it takes. The dt for each trajectory is given in the geom_config.geom_config_extension_real_trajectory object. This simulation corresponds to experiment {} with the odor state: {}.'

# ENVIRONMENT
DT = -1  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)

# PLUME
PLUME_PARAMS_DICT = {
    'fruitfly_0.3mps_checkerboard_floor': {
        'max_conc': 194.4013571501637,
        'threshold': 10,
        'ymean': -0.01686711918990497,
        'zmean': -0.024448968890361408,
        'ystd': 0.013907665641150927,
        'zstd': 0.024051865551085332
    },
    'fruitfly_0.4mps_checkerboard_floor': {
        'max_conc': 526.5611158275608,
        'threshold': 10,
        'ymean': 0.010397528868787908,
        'zmean': 0.021053005635139854,
        'ystd': 0.011163543244851111,
        'zstd': 0.012783432241805101
    },
    'fruitfly_0.6mps_checkerboard_floor': {
        'max_conc': 470.45545552729538,
        'threshold': 10,
        'ymean': 0.0046019902473019899,
        'zmean': 0.051356187443552505,
        'ystd': 0.0084829247732408405,
        'zstd': -0.0091643524066678372
    },
}

# INSECT
INSECT_PARAMS_DICT = {
    'fruitfly_0.3mps_checkerboard_floor': {
        'w': 0.3,  # wind (m/s)
        'r': 1000,  # source emission rate
        'd': 0.12,  # diffusivity (m^2/s)
        'a': .002,  # searcher size (m)
        'tau': 10000,  # particle lifetime (s)
    },
    'fruitfly_0.4mps_checkerboard_floor': {
        'w': 0.4,  # wind (m/s)
        'r': 1000,  # source emission rate
        'd': 0.12,  # diffusivity (m^2/s)
        'a': .002,  # searcher size (m)
        'tau': 10000,  # particle lifetime (s)
    },
    'fruitfly_0.6mps_checkerboard_floor': {
        'w': 0.6,  # wind (m/s)
        'r': 1000,  # source emission rate
        'd': 0.12,  # diffusivity (m^2/s)
        'a': .002,  # searcher size (m)
        'tau': 10000,  # particle lifetime (s)
    }
}
LOGLIKE = binary_advec_diff_tavg