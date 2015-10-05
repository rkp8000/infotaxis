__author__ = 'rkp'

import os

REPO_DIR = os.getenv('REPOSITORY_DIRECTORY')
WT_REPO = os.path.join(REPO_DIR, 'wind_tunnel')

EXPERIMENTS = ('fruitfly_0.3mps_checkerboard_floor',
               'fruitfly_0.4mps_checkerboard_floor',
               'fruitfly_0.6mps_checkerboard_floor',
               'mosquito_0.4mps_checkerboard_floor')
ODOR_STATES = ('on', 'none', 'afterodor')