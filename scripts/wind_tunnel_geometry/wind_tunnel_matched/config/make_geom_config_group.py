import numpy as np

from plume import Environment3d

SCRIPTID = 'make_geom_config_group_wind_tunnel_matched'

EXPERIMENTID = 'fruitfly_0.3mps_checkerboard_floor'
ODORSTATES = ['on', 'none', 'afterodor']

GEOMCONFIGGROUPID = 'wind_tunnel_matched_' + EXPERIMENTID
GEOMCONFIGGROUPDESCRIPTION = ('All starting position indexes and durations for experiment "{}"'
                              ' with odor "{}", clean trajectories only.')

# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)