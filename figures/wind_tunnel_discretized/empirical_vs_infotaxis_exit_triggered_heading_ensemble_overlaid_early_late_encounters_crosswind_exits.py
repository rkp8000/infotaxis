"""
Plot the plume-crossing-triggered average, standard deviation, and standard error of the mean for
empirical trajectories and matched infotaxis trajectories. This plot overlays results from early and late encounter sets.
"""
from __future__ import division, print_function

COLORS = {'early': 'k',
          'late': 'r'}
WIND_SPEEDS = ('0.3 m/s', '0.4 m/s', '0.6 m/s')

import numpy as np
import matplotlib.pyplot as plt

from math_tools.plot import set_fontsize

from db_api import models
from db_api.connect import session

from config import *
from config.exit_triggered_ensemble import *

# parameters for this plot

FACE_COLOR = 'white'
FIG_SIZE = (16, 10)
FONT_SIZE = 20
X_LIM = 0, 30
Y_LIM = 40, 150


figs = []
axss = []
for _ in range(3):
    fig, axs = plt.subplots(2, 3, facecolor=FACE_COLOR, figsize=FIG_SIZE, tight_layout=True)
    figs += [fig]
    axss += [axs]

for s_ctr, sg_id_template in enumerate((SEGMENT_GROUP_ID_EMPIRICAL, SEGMENT_GROUP_ID_INFOTAXIS)):
    for e_ctr, expt in enumerate(EXPERIMENTS):
        for o_ctr, odor_state in enumerate(ODOR_STATES):
            for c in ('early', 'late'):
                sg_id = sg_id_template.format(expt, odor_state)
                seg_group = session.query(models.SegmentGroup).get(sg_id)

                heading_ensemble = None
                for ens in seg_group.analysis_triggered_ensembles:
                    # only get the heading ensemble if it has correct conditions
                    conditions = (ens.variable == 'heading',
                                  ens.trigger_start == 'exit',
                                  59 < ens.heading_min < 61,
                                  119 < ens.heading_max < 121,
                                  ens.x_idx_max == 50,
                                  ens.x_idx_min == 15)

                    if c == 'early':
                        conditions += (ens.encounter_number_min == 1,
                                       ens.encounter_number_max == 2,)
                    elif c == 'late':
                        conditions += (ens.encounter_number_min == 3,
                                       ens.encounter_number_max == None,)

                    if all(conditions):
                        heading_ensemble = ens
                        break

                heading_ensemble.fetch_data(session)
                if heading_ensemble._data is None:
                    continue

                time_vector = np.arange(len(heading_ensemble.mean))
                axs = axss[e_ctr]
                ax = axs[s_ctr, o_ctr]
                ax.errorbar(time_vector, heading_ensemble.mean, lw=3,
                            yerr=heading_ensemble.sem, color=COLORS[c])
                ax.set_xlim(X_LIM)
                ax.set_ylim(Y_LIM)

                if s_ctr == o_ctr == 0:
                    ax.legend(['early', 'late'], fontsize=FONT_SIZE)
                if s_ctr == 0 and o_ctr == 1:
                    ax.set_title('wind {} \n empirical \n'.format(WIND_SPEEDS[e_ctr]))
                elif s_ctr == 1 and o_ctr == 1:
                    ax.set_title('infotaxis \n'.format(WIND_SPEEDS[e_ctr]))

                if s_ctr == 1:
                    ax.set_xlabel('timesteps')
                if o_ctr == 0:
                    ax.set_ylabel('heading (deg)')


for axs in axss:
    [set_fontsize(ax, FONT_SIZE) for ax in axs.flatten()]

plt.show(block=True)