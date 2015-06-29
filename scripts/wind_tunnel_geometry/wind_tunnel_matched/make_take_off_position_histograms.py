"""
Create 3d histograms of take-off positions for empirical data set (though these are the same as the infotaxis dataset by construction).
"""
from __future__ import division, print_function

SCRIPT_ID = 'make_take_off_position_histograms'
SCRIPT_NOTES = 'Run for all experiments and all odor states for wind tunnel discretized copies.'

import numpy as np

from db_api.connect import session
from db_api import models, add_script_execution

# get configuration
from config import *
from config.make_take_off_position_histograms import *


def main():

    add_script_execution(SCRIPT_ID, session=session, notes=SCRIPT_NOTES)

    for expt in EXPERIMENTS:
        for odor_state in ODOR_STATES:

            sim_id = SIMULATION_ID.format(expt, odor_state)
            sim = session.query(models.Simulation).get(sim_id)

            print(sim_id)

            pos_idxs_start = []

            for trial in sim.trials:
                tp_id_start = trial.start_timepoint_id
                tp = session.query(models.Timepoint).get(tp_id_start)
                pos_idxs_start += [(tp.xidx, tp.yidx, tp.zidx)]

            pos_start = [sim.env.pos_from_idx(idx) for idx in pos_idxs_start]

            # build the histogram
            bins = (sim.env.xbins, sim.env.ybins, sim.env.zbins)
            hist, _ = np.histogramdd(np.array(pos_start), bins=bins)

            # create the data model and store it
            hist_data_model = models.SimulationAnalysisTakeOffPositionHistogram()
            hist_data_model.simulation = sim
            hist_data_model.store_data(session, hist.astype(int))

            session.add(hist_data_model)

    session.commit()

if __name__ == '__main__':
    main()