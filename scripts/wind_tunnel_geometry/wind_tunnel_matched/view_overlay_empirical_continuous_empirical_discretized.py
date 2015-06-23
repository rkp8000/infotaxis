"""
Plot continuous empirical trajectories overlaid with discretized versions.

author: @rkp
"""
from __future__ import print_function, division

import os
import imp
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from db_api.connect import session
from db_api import models
from plotting import multi_traj_3d as plot_multi_traj

from plume import CollimatedPlume

# get configuration
from config import *
from config.view_overlay_empirical_continuous_empirical_discretized import *


# get wind tunnel connection and models
wt_session = imp.load_source('connect', os.path.join(WT_REPO, 'db_api', 'connect.py')).session
wt_models = imp.load_source('models', os.path.join(WT_REPO, 'db_api', 'models.py'))

# get simulation
sim = session.query(models.Simulation).get(SIMULATION_ID)

fig, axs = plt.subplots(2, 1)

for trial in sim.trials:
    [ax.cla() for ax in axs]

    # get discrete positions from trial
    timepoints_discrete = trial.get_timepoints(session)
    positions_discrete = [sim.env.pos_from_idx((tp.xidx, tp.yidx, tp.zidx))
                          for tp in timepoints_discrete]
    positions_discrete = np.array(positions_discrete)

    # get corresponding trajectory from wind tunnel database
    traj_id = trial.geom_config.extension_real_trajectory.real_trajectory_id
    traj = wt_session.query(wt_models.Trajectory).get(traj_id)

    # get continuous positions that were discretized
    timepoints_continuous = traj.get_timepoints(wt_session)
    positions_continuous = [(tp.x, tp.y, tp.z) for tp in timepoints_continuous]
    positions_continuous = np.array(positions_continuous)

    # get plume for background
    pl = CollimatedPlume(env=sim.env, orm=sim.plume)
    pl.initialize()

    plot_multi_traj(axs=axs, env=sim.env, bkgd=[pl.concxy, pl.concxz],
                    trajs=[positions_continuous, positions_discrete])

    plt.draw()
    print(traj_id)
    raw_input()

plt.show(block=True)