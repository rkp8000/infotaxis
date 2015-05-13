from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from db_api.mappings import Simulation
from db_api.connect import engine, session

from config.wind_tunnel_basic_plume_vary_R import *


# get data
sims = session.query(Simulation).filter(Simulation.id.like(SIMULATIONIDTEMPLATE)).all()

prob_found_src_dict = {}
mean_search_time_dict = {}
std_search_time_dict = {}

for sim in sims:
    # get r
    pps = sim.plume.plume_params
    r = [pp.value for pp in pps if pp.name=='r'][0]

    trial_infos = [trial.trial_info for trial in sim.trials]

    prob_found_src = np.sum([ti.found_src for ti in trial_infos]) / len(trial_infos)
    prob_found_src_dict[r] = prob_found_src

    search_times = [ti.duration for ti in trial_infos if ti.found_src]
    mean_search_time_dict[r] = np.mean(search_times)
    std_search_time_dict[r] = np.std(search_times)


# make figure
plt.ion()
fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].scatter(prob_found_src_dict.keys(), prob_found_src_dict.values())
axs[0].set_ylabel('P(source found)')

axs[1].errorbar(mean_search_time_dict.keys(), mean_search_time_dict.values(), yerr=std_search_time_dict.values(), fmt='o')
axs[1].set_xlabel('source emission rate')
axs[1].set_ylabel('Mean search time (time steps)')

# show figure
plt.draw()
plt.show()