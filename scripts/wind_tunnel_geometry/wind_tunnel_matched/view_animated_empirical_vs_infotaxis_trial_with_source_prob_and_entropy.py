"""
Select a specific empirical trial and its matched infotaxis trial and animate them on top of their corresponding source probability distributions.
"""
from __future__ import print_function, division

TRIAL_ID_EMPIRICAL = 26
SIMULATION_ID_EMPIRICAL = 'wind_tunnel_discretized_copies_fruitfly_0.4mps_checkerboard_floor_odor_afterodor'
SIMULATION_ID_INFOTAXIS = 'wind_tunnel_discretized_matched_r1000_d0.12_fruitfly_0.4mps_checkerboard_floor_odor_afterodor'

FIG_SIZE = (16, 6)
PAUSE_EVERY = 10

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.ion()

from db_api import models
from db_api.connect import session

import insect
import plume
from logprob_odor import binary_advec_diff_tavg
from trial import Trial, TrialFromPositionSequence

from plotting import multi_traj_3d_with_entropy as plot_traj

sim_empirical = session.query(models.Simulation).get(SIMULATION_ID_EMPIRICAL)
sim_infotaxis = session.query(models.Simulation).get(SIMULATION_ID_INFOTAXIS)
trial_empirical = session.query(models.Trial).get(TRIAL_ID_EMPIRICAL)

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

timepoints_infotaxis = trial_infotaxis.get_timepoints(session)
positions_infotaxis = [sim_infotaxis.env.pos_from_idx((tp.xidx, tp.yidx, tp.zidx,))
                       for tp in timepoints_infotaxis]
positions_infotaxis = np.array(positions_infotaxis)

duration = len(positions_infotaxis)

# create empirical and infotaxis insects
gc = trial_empirical.geom_config
dt = gc.extension_real_trajectory.avg_dt
ins_empirical = insect.Insect(env=sim_empirical.env, dt=dt, orm=sim_empirical.insect)
ins_infotaxis = insect.Insect(env=sim_infotaxis.env, dt=dt, orm=sim_infotaxis.insect)
ins_empirical.loglike_function = binary_advec_diff_tavg
ins_infotaxis.loglike_function = binary_advec_diff_tavg
ins_infotaxis.set_pos(gc.start_idx, is_idx=True)
ins_infotaxis.initialize()


# create plumes
pl_empirical = plume.EmptyPlume(env=sim_empirical.env, dt=0)
pl_empirical.initialize()

pl_infotaxis = plume.EmptyPlume(env=sim_infotaxis.env, dt=0)
pl_infotaxis.initialize()

# create new trial for both infotaxis and empirical
trial_new_empirical = TrialFromPositionSequence(positions_empirical,
                                                pl=pl_empirical,
                                                ins=ins_empirical,
                                                already_discretized=True,
                                                run_all_steps=False)

trial_new_infotaxis = Trial(pl=pl_infotaxis, ins=ins_infotaxis, nsteps=duration)

print(ins_infotaxis.params)
print(ins_empirical.params)

# run trials and plot them
fig, axs = plt.subplots(3, 2, facecolor='white', figsize=FIG_SIZE)

for ts in range(duration):

    [ax.cla() for ax in axs.flatten()]

    if ts > 0:
        trial_new_empirical.step()
        trial_new_infotaxis.step()

    src_probs_empirical = (trial_new_empirical.ins.logprobxy, trial_new_empirical.ins.logprobxz)
    s_empirical = stats.entropy(trial_new_empirical.ins.get_src_prob(log=False, normalized=True).flatten())

    src_probs_infotaxis = (trial_new_infotaxis.ins.logprobxy, trial_new_infotaxis.ins.logprobxz)
    s_infotaxis = stats.entropy(trial_new_infotaxis.ins.get_src_prob(log=False, normalized=True).flatten())

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
    if ts and (ts % PAUSE_EVERY == 0):
        print('s_empirical: {}'.format(s_empirical))
        print('s_infotaxis: {}'.format(s_infotaxis))
        import pdb; pdb.set_trace()

plt.show(block=True)