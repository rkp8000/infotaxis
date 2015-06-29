"""
Plot the trajectory total displacement heatmaps projected for the discretized empirical trajectories.
"""
from __future__ import print_function, division

FIG_SIZE = (16, 7)
FONT_SIZE = 20
PROJECTION = 'xy'

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from db_api import models
from db_api.connect import session

from config import *
from config.position_heatmap import *

row_labels = ('0.3 m/s', '0.4 m/s', '0.6 m/s')
col_labels = ('on', 'none', 'afterodor')

for sim_id_template in (SIMULATION_ID_EMPIRICAL, SIMULATION_ID_INFOTAXIS):
    fig, axs = plt.subplots(3, 3, facecolor='white', figsize=FIG_SIZE, tight_layout=True)

    for e_ctr, expt in enumerate(EXPERIMENTS):
        for o_ctr, odor_state in enumerate(ODOR_STATES):

            sim_id = sim_id_template.format(expt, odor_state)
            sim = session.query(models.Simulation).get(sim_id)

            sim.analysis_displacement_total_histogram.fetch_data(session)

            if PROJECTION == 'xy':
                heatmap = sim.analysis_displacement_total_histogram.xy
                extent = sim.env.extentxy
                xlabel = 'x'
                ylabel = 'y'
            elif PROJECTION == 'xz':
                heatmap = sim.analysis_displacement_total_histogram.xz
                extent = sim.env.extentxz
                xlabel = 'x'
                ylabel = 'z'
            elif PROJECTION == 'yz':
                heatmap = sim.analysis_displacement_total_histogram.yz
                extent = sim.env.extentyz
                xlabel = 'y'
                ylabel = 'z'

            # calculate entropy
            entropy = stats.entropy((heatmap / heatmap.sum()).flatten())
            ax = axs[e_ctr, o_ctr]
            ax.matshow(np.log(heatmap).T, origin='lower', extent=extent)

            # labels
            if e_ctr == 2:
                ax.set_xlabel(xlabel)

            if o_ctr == 0:
                ax.set_ylabel(ylabel)

            ax.set_title('{} {}\nS = {}'.format(row_labels[e_ctr], col_labels[o_ctr], entropy))

    if sim_id_template == SIMULATION_ID_EMPIRICAL:
        fig.suptitle('empirical\n', fontsize=FONT_SIZE)
    elif sim_id_template == SIMULATION_ID_INFOTAXIS:
        fig.suptitle('infotaxis\n', fontsize=FONT_SIZE)

plt.show(block=True)