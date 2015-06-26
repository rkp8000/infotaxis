"""
Create 3d histograms of total displacement from the beginning to the end of the trajectory for the specified simulations, experiments, and odor_states. Save them in the database.
"""

SCRIPTID = 'generate_displacement_total_histograms'
SCRIPTNOTES = 'Run for all experiments and all odor states for wind tunnel discretized copies and wind tunnel matched infotaxis trajectories.'

import numpy as np

from db_api import models
from db_api.connect import session
from db_api import add_script_execution

from config import *
from config.make_displacement_total_histograms_3d import *


def main(traj_limit=None):
    # add script execution to database
    add_script_execution(SCRIPTID, session=session, multi_use=False, notes=SCRIPTNOTES)

    for sim_id_template in SIMULATION_IDS:
        for expt in EXPERIMENTS:
            for odor_state in ODOR_STATES:

                sim_id = sim_id_template.format(expt, odor_state)

                print(sim_id)

                sim = session.query(models.Simulation).get(sim_id)

                # get the displacements for all trials
                displacements = []
                for trial in sim.trials[:traj_limit]:
                    tps = trial.get_timepoints(session).all()
                    pos_idx_start = np.array((tps[0].xidx, tps[0].yidx, tps[0].zidx))
                    pos_idx_end = np.array((tps[-1].xidx, tps[-1].yidx, tps[-1].zidx))
                    displacements += (pos_idx_end - pos_idx_start).astype(int)

                # build the histogram
                x_bins = np.arange(-sim.env.nx, sim.env.nx) + 0.5
                y_bins = np.arange(-sim.env.ny, sim.env.ny) + 0.5
                z_bins = np.arange(-sim.env.nz, sim.env.nz) + 0.5

                displacement_histogram, _ = \
                    np.histogramdd(displacements, bins=(x_bins, y_bins, z_bins))

                # create the data model and store it
                displacement_hist_data_model = \
                    models.SimulationAnalysisDisplacementTotalHistogram()
                displacement_hist_data_model.simulation = sim

                displacement_hist_data_model. \
                    store_data(session, displacement_hist_data_model.astype(int))
                session.add(displacement_hist_data_model)

                session.commit()


if __name__ == '__main__':
    main()