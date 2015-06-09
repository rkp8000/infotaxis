"""
Plot continuous empirical trajectories overlaid with discretized versions.

author: @rkp
"""
from __future__ import print_function, division

import os
import imp
import numpy as np
import matplotlib.pyplot as plt

from db_api.connect import session
from db_api import models
from plotting import multi_traj as plot_multi_traj

# get configuration
from config.view_overlay_empirical_continuous_empirical_discretized import *


# get wind tunnel connection and models
wt_session = imp.load_source('connect', os.path.join(WT_REPO, 'db_api', 'connect.py')).session
wt_models = imp.load_source('models', os.path.join(WT_REPO, 'db_api', 'models.py'))

# get simulation
sim = session.query(models.Simulation).get(SIMULATION_ID)

fig, axs = plt.subplots(2, 1)

for trial in sim.trials:

    timepoints_discrete = trial.get_timepoints(session)

    # get corresponding trajectory from wind tunnel database
    traj_id = trial.geom_config.extension_real_trajectory.real_trajectory_id
    traj = wt_session.query(wt_models.Trajectory)

    timepoints_continuous = traj.get_timepoints(wt_session)

    plot_multi_traj(...)