"""
Plot continuous empirical trajectories overlaid with discretized versions.

author: @rkp
"""
from __future__ import print_function, division

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
from config.view_overlay_empirical_discretized_infotaxis import *


# get wind tunnel connection and models
wt_session = imp.load_source('connect', os.path.join(WT_REPO, 'db_api', 'connect.py',)).session
wt_models = imp.load_source('models', os.path.join(WT_REPO, 'db_api', 'models.py',))

# get simulation
sim = session.query(models.Simulation).get(SIMULATION_ID_EMPIRICAL)

fig, axs = plt.subplots(2, 1)

for trial_empirical in sim.trials:
    [ax.cla() for ax in axs]

    # get positions from empirical trial
    timepoints_discrete = trial_empirical.get_timepoints(session)
    positions_discrete = [sim.env.pos_from_idx((tp.xidx, tp.yidx, tp.zidx,))
                          for tp in timepoints_discrete]
    positions_discrete_empirical = np.array(positions_discrete)

    # find infotaxis trial
    trial_infotaxis = None
    for trial in trial_empirical.geom_config.trials:
        if trial.simulation.id == SIMULATION_ID_INFOTAXIS:
            trial_infotaxis = trial
            break

    # get positions from infotaxis trial
    timepoints_discrete = trial_infotaxis.get_timepoints(session)
    positions_discrete = [sim.env.pos_from_idx((tp.xidx, tp.yidx, tp.zidx,))
                          for tp in timepoints_discrete]
    positions_discrete_infotaxis = np.array(positions_discrete)

    # get plume for background
    pl = CollimatedPlume(env=sim.env, orm=sim.plume)
    pl.initialize()

    plot_multi_traj(axs=axs, env=sim.env, bkgd=[pl.concxy, pl.concxz, ],
                    trajs=[positions_discrete_empirical, positions_discrete_infotaxis, ])

    plt.draw()
    raw_input()

plt.show(block=True)