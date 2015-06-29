"""
Plot the plume-crossing-triggered average, standard deviation, and standard error of the mean for empirical trajectories and matched infotaxis trajectories. This plot has no conditions on any of the trajectory segments.
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from db_api import models
from db_api.connect import session

from config import *
from config.exit_triggered_ensemble import *


fig, axs = plt.subplots(2, 3)

for s_ctr, sim_id_template in enumerate((SIMULATION_ID_EMPIRICAL, SIMULATION_ID_INFOTAXIS)):
    for e_ctr, expt in enumerate(EXPERIMENTS):
        for o_ctr, odor_state in enumerate(ODOR_STATES):

            sim_id = sim_id_template.format(expt, odor_state)
            sim = session.query(models.Simulation).get(sim_id)

            heading_ensemble = None
            for h in sim.analysis_exit_triggered_heading_ensembles:
                # only get the heading ensemble if it has no conditions
                if not h.conditions:
                    heading_ensemble = h
                    break

            t = heading_ensemble.time_vector
            heading_ensemble.fetch_data(session)

            axs[s_ctr, o_ctr].errorbar(t, heading_ensemble.mean, yerr=heading_ensemble.std,
                                       color=COLORS_EXPT[expt])

