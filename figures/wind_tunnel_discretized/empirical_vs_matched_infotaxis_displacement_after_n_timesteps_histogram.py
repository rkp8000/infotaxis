from __future__ import division, print_function

PROJECTION = 'xy'
FIG_SIZE = (16, 7)
FONT_SIZE = 20
N_TIMESTEPS = 1  # 1, 2, 5, 10, 20, or 50

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

            # get histogram
            hist = None
            for h in sim.analysis_displacement_after_n_timesteps_histograms:
                if h.n_timesteps == N_TIMESTEPS:
                    hist = h
                    break

            hist.fetch_data(session)

            if PROJECTION == 'xy':
                heatmap = hist.xy
                extent = hist.extent_xy
                xlabel = 'x'
                ylabel = 'y'
            elif PROJECTION == 'xz':
                heatmap = hist.xz
                extent = hist.extent_xz
                xlabel = 'x'
                ylabel = 'z'
            elif PROJECTION == 'yz':
                heatmap = hist.yz
                extent = hist.extent_yz
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

            ax.set_title('{} {}\nS = {}\n'.format(row_labels[e_ctr], col_labels[o_ctr], entropy))

    if sim_id_template == SIMULATION_ID_EMPIRICAL:
        fig.suptitle('empirical\n', fontsize=FONT_SIZE)
    elif sim_id_template == SIMULATION_ID_INFOTAXIS:
        fig.suptitle('infotaxis\n', fontsize=FONT_SIZE)

plt.show(block=True)