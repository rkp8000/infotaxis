import numpy as np

from plume import Environment3d

EXPERIMENTIDS = ('fruitfly_0.3mps_checkerboard_floor',
                 'fruitfly_0.4mps_checkerboard_floor',
                 'fruitfly_0.6mps_checkerboard_floor',)
ODORSTATES = ('on', 'none', 'afterodor')

GEOMCONFIGGROUPIDS = ['wind_tunnel_matched_' + EXPERIMENTID for EXPERIMENTID in EXPERIMENTIDS]

# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)