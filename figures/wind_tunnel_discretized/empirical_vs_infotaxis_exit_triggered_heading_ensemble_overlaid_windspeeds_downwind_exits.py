"""
Plot the plume-crossing-triggered average, standard deviation, and standard error of the mean for
empirical trajectories and matched infotaxis trajectories. This plot uses only segments at which
the agent exited the plume with a downwind heading.
"""
from __future__ import division, print_function

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
Y_LIM = 50, 170


fig, axs = plt.subplots(2, 3, facecolor='white', figsize=FIG_SIZE, tight_layout=True)

for s_ctr, sg_id_template in enumerate((SEGMENT_GROUP_ID_EMPIRICAL, SEGMENT_GROUP_ID_INFOTAXIS)):
    for e_ctr, expt in enumerate(EXPERIMENTS):
        for o_ctr, odor_state in enumerate(ODOR_STATES):

            sg_id = sg_id_template.format(expt, odor_state)
            seg_group = session.query(models.SegmentGroup).get(sg_id)

            heading_ensemble = None
            for ens in seg_group.analysis_triggered_ensembles:
                # only get the heading ensemble if it has correct conditions
                conditions = (ens.variable == 'heading',
                              ens.trigger_start == 'exit',
                              119 < ens.heading_min < 121,
                              179 < ens.heading_max < 181,
                              ens.encounter_number_min is None,
                              ens.encounter_number_max is None)

                if all(conditions):
                    heading_ensemble = ens
                    break

            heading_ensemble.fetch_data(session)
            if heading_ensemble._data is None:
                continue

            time_vector = np.arange(len(heading_ensemble.mean))

            ax = axs[s_ctr, o_ctr]
            ax.errorbar(time_vector, heading_ensemble.mean, lw=3,
                        yerr=heading_ensemble.sem, color=COLORS_EXPT[expt])

            ax.set_ylim(Y_LIM)
            ax.set_xlim(X_LIM)

            if s_ctr == 1:
                ax.set_xlabel('timesteps')
            if o_ctr == 0:
                ax.set_ylabel('heading (degree)')

            if s_ctr == 0 and o_ctr == 1:
                ax.set_title('empirical')
            if s_ctr == 1 and o_ctr == 1:
                ax.set_title('infotaxis')


for ax in axs.flatten():
    set_fontsize(ax, FONT_SIZE)

axs[0, 0].legend(['0.3 m/s', '0.4 m/s', '0.6 m/s'], fontsize=24)

plt.show(block=True)