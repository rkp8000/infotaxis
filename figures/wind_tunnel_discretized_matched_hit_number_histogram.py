"""
Plot a histogram of the number of hits received by the wind-tunnel-discretized-matched
infotaxis trajectories for all experiments.
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from db_api.connect import session
from db_api import models

from config.wind_tunnel_discretized_matched_hit_number_histogram import *


hit_numbers = {expt: {} for expt in EXPERIMENTS}

for expt in EXPERIMENTS:
    for odor_state in ODOR_STATES:
        # get all trials for the simulation from this odor state and experiment
        sim_id = SIMULATION_ID.format(expt, odor_state)
        print('Getting hit number distribution from simulation "{}" ...'.format(sim_id))

        trials = session.query(models.Trial).filter_by(simulation_id=sim_id)

        # get the number of hits in each trial
        hits_per_trial = []
        for trial in trials:

            n_hits = np.sum([tp.detected_odor for tp in trial.get_timepoints(session)])
            hits_per_trial += [n_hits]

        hit_numbers[expt][odor_state] = hits_per_trial


# plot hit number histogram for each experiment and odor_state
fig, axs = plt.subplots(3, 3, facecolor='white', tight_layout=True)

for e_ctr, expt in enumerate(EXPERIMENTS):
    for o_ctr, odor_state in enumerate(ODOR_STATES):

        axs[e_ctr, o_ctr].hist(hit_numbers[expt][odor_state], normed=True)

[ax.set_xlabel('number of hits') for ax in axs[-1, :]]
[ax.set_ylabel('probability') for ax in axs[:, 0]]

plt.show(block=True)