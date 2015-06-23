"""
Plot the position heatmaps projected onto the xy-plane for the discretized empirical trajectories.
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt

from db_api import models
from db_api.connect import session

from config import *
from config.empirical_position_heatmap_xy import *


fig, axs = plt.subplots(3, 3, facecolor='white', tight_layout=True)

for e_ctr, expt in enumerate(EXPERIMENTS):
    for o_ctr, odor_state in enumerate(ODOR_STATES):

        sim_id = SIMULATION_ID.format(expt, odor_state)
        sim = session.query(models.Simulation).get(sim_id)

        sim.analysis_position_histogram.fetch_data(session)
        heatmap_xy = sim.analysis_position_histogram.xy

        axs[e_ctr, o_ctr].matshow(heatmap_xy.T, origin='lower', extent=sim.env.extentxy)

plt.show(block=True)