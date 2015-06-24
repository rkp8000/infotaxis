"""
Select a specific empirical trial and its matched infotaxis trial and animate them on top of their corresponding source probability distributions.
"""
from __future__ import print_function, division

TRIAL_ID = 100
SIMULATION_ID_EMPIRICAL = 'blah'
SIMULATION_ID_INFOTAXIS = 'blah'

FIG_SIZE = (16, 6)

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from db_api import models
from db_api.connect import session

import insect
import plume
import trial

from plotting import multi_traj_3d_with_entropy as plot_traj

sim_empirical = session.query(models.Simulation).get(SIMULATION_ID_EMPIRICAL)
sim_infotaxis = session.query(models.Simulation).get(SIMULATION_ID_INFOTAXIS)
trial_empirical = session.query(models.Trial).filter(models.Trial.simulation == sim_empirical)

# get infotaxis trial
trial_infotaxis = None
for trial in trial_empirical.geom_config.trials:
    if trial.simulation.id == SIMULATION_ID_INFOTAXIS:
        trial_infotaxis = trial
        break

# get positions for both of these
timepoints_empirical = trial_empirical.get_timepoints(session)
positions_empirical = [sim_empirical.env.pos_from_idx((tp.xidx, tp.yidx, tp.zidx,))
                       for tp in timepoints_empirical]
positions_empirical = np.array(positions_empirical)

timepoints_infotaxis = trial_empirical.get_timepoints(session)
positions_infotaxis = [sim_infotaxis.env.pos_from_idx((tp.xidx, tp.yidx, tp.zidx,))
                       for tp in timepoints_infotaxis]
positions_infotaxis = np.array(positions_infotaxis)

duration = len(positions_infotaxis)

# create empirical and infotaxis insects
dt = trial_empirical.geom_config.extension_real_trajectory.avg_dt
ins_empirical = insect.Insect(env=sim_empirical.env, dt=dt, orm=sim_empirical.insect)
ins_infotaxis = insect.Insect(env=sim_infotaxis.env, dt=dt, orm=sim_infotaxis.insect)

# create plumes
pl_empirical = plume.EmptyPlume(env=sim_empirical.env, dt=0)
pl_empirical.initialize()

pl_infotaxis = plume.EmptyPlume(env=sim_infotaxis.env, dt=0)
pl_infotaxis.initialize()

# create new trial for both infotaxis and empirical
trial_new_empirical = trial.TrialFromPositionSequence(positions_empirical,
                                                      pl=pl_empirical,
                                                      ins=ins_empirical,
                                                      run_all_steps=False)

trial_new_infotaxis = trial.Trial(pl=pl_infotaxis, ins=ins_infotaxis, nsteps=duration)

# run trials and plot them
fig, axs = plt.subplots(2, 2, facecolor='white', figsize=FIG_SIZE)

for ts in range(duration):

    [ax.cla() for ax in axs.flatten()]

    if ts > 0:
        trial_new_empirical.step()
        trial_new_infotaxis.step()

    src_probs_empirical = (trial_new_empirical.ins.logprob_xy, trial_new_empirical.ins.logprob_xz)

    src_probs_infotaxis = (trial_new_infotaxis.ins.logprob_xy, trial_new_infotaxis.ins.logprob_xz)

    traj_empirical = trial_new_empirical.pos[:trial_new_empirical.ts + 1, :]
    traj_infotaxis = trial_new_infotaxis.pos[:trial_new_infotaxis.ts + 1, :]

    entropy_empirical = trial_new_empirical.entropies[:trial_new_empirical.ts + 1]
    entropy_infotaxis = trial_new_infotaxis.entropies[:trial_new_infotaxis.ts + 1]

    # plot both trajectories
    plot_traj(axs[:, 0], sim_empirical.env, bkgd=src_probs_empirical, trajs=[traj_empirical],
              entropies=[entropy_empirical])
    plot_traj(axs[:, 1], sim_infotaxis.env, bkgd=src_probs_infotaxis, trajs=[traj_infotaxis],
              entropies=[entropy_infotaxis])

    plt.draw()
    raw_input()