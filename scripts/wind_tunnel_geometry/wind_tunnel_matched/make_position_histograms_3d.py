"""
Create 3d position histograms for the specified simulations, experiments, and odor_states. Save them in the database.
"""

import numpy as np

from db_api import models
from db_api.connect import session

from config import *
from config.make_position_histograms_3d import *


def main(traj_limit=None):

    for sim_id_template in SIMULATION_IDS:
        for expt in EXPERIMENTS:
            for odor_state in ODOR_STATES:

                sim_id = sim_id_template.format(expt, odor_state)

                sim = session.query(models.Simulation).get(sim_id)

                # get the position indexes for all time points for all trials
                pos_idxs = []
                for trial in sim.trials[:traj_limit]:
                    tps = trial.get_timepoints(session)
                    pos_idxs += [np.array([(tp.xidx, tp.yidx, tp.zidx) for tp in tps])]

                pos_idxs = np.concatenate(pos_idxs, axis=0)

                # build the histogram
                bins = (sim.env.xbins, sim.env.ybins, sim.env.zbins)
                pos_histogram = np.histogramdd(pos_idxs, bins=bins)

                # create the data model and store it
                pos_hist_data_model = models.SimulationAnalysisPositionHistogram()
                pos_hist_data_model.sim = sim

                pos_hist_data_model.store_data(session, pos_histogram)
                session.add(pos_hist_data_model)

                session.commit()

if __name__ == '__main__':
    main()