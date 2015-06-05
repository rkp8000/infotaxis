import os
import numpy as np

from plume import Environment3d

# LOCAL PATHS
WT_REPO = os.path.join(os.getenv('REPOSITORY_DIRECTORY'), 'wind_tunnel')

# THINGS TO LOOP OVER
EXPERIMENT_IDS = ('fruitfly_0.3mps_checkerboard_floor',
                  'fruitfly_0.4mps_checkerboard_floor',
                  'fruitfly_0.6mps_checkerboard_floor',)
ODOR_STATES = ('on', 'none', 'afterodor')

# GEOM CONFIG GROUP
GEOM_CONFIG_GROUP_ID = 'wind_tunnel_matched_discretized'
GEOM_CONFIG_GROUP_DESC = 'Geometrical configurations derived from wind tunnel trajectories by discretizing them according to the Environment3d and Plume objects. Trajectory durations are based on the number of grid points in the discretization that the insect crossed, not the total flight time in seconds. This group corresponds to experiment {} with the odor state: {}.'

# ENVIRONMENT
DT = -1 # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)