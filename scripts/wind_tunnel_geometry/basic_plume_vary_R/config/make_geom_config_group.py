import numpy as np
from plume import Environment3d

ID = 'wind_tunnel_random_src_and_start'
DESCRIPTION = 'Random source and start positions using wind tunnel geometry. Max durations all 1000 timesteps.'
DURATION = 1000

NGEOMCONFIGS = 200

# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 16)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)


