"""
Plot continuous empirical trajectories overlaid with discretized versions.

author: @rkp
"""
from __future__ import print_function, division

TRIAL_ID = 4

import numpy as np
import matplotlib.pyplot as plt

from db_api.connect import session
from db_api import models
from plotting import traj_3d_with_segments_and_entropy as plot_traj

from plume import CollimatedPlume

# get trial
trial = session.query(models.Trial).get(TRIAL_ID)
sim = trial.simulation

# get positions from empirical trial
timepoints = trial.get_timepoints(session)
positions = [sim.env.pos_from_idx((tp.xidx, tp.yidx, tp.zidx,))
             for tp in timepoints]
positions = np.array(positions)
detected_odors = np.array([tp.detected_odor for tp in timepoints])
entropies = np.array([tp.src_entropy for tp in timepoints])

# get segment starts and ends
timepoint_start = trial.start_timepoint_id
seg_starts = [segment.timepoint_id_enter - timepoint_start for segment in trial.segments]
seg_ends = [segment.timepoint_id_exit - timepoint_start + 1 for segment in trial.segments]


# get plume for background
pl = CollimatedPlume(env=sim.env, orm=sim.plume)
pl.initialize()

fig, axs = plt.subplots(3, 1)
plot_traj(axs=axs, env=sim.env, bkgd=[pl.concxy, pl.concxz, ],
          traj=positions, entropies=entropies, seg_starts=seg_starts, seg_ends=seg_ends)

print('From simulation "{}"'.format(sim.id))
plt.show(block=True)