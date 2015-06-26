"""
Create 3d histograms of displacement from the beginning of the trajectory to a point after a certain number of timesteps have gone by for the specified simulations. Save them in the database.
"""

SCRIPTID = 'make_displacement_after_n_timesteps_histograms'
SCRIPTNOTES = 'Run for all experiments and all odor states for wind tunnel discretized copies and wind tunnel matched infotaxis trajectories. Run for [1, 2, 5, 10, 20, 50] timesteps.'

import numpy as np

from db_api import models
from db_api.connect import session
from db_api import add_script_execution

from config import *
from config.make_displacement_after_n_timesteps_histograms_3d import *


def main(traj_limit=None):
    # add script execution to database
    add_script_execution(SCRIPTID, session=session, multi_use=False, notes=SCRIPTNOTES)

    for sim_id_template in SIMULATION_IDS:
        for expt in EXPERIMENTS:
            for odor_state in ODOR_STATES:

                sim_id = sim_id_template.format(expt, odor_state)

                print(sim_id)

                sim = session.query(models.Simulation).get(sim_id)

                for n_timesteps in N_TIMESTEPSS:
                    # get the displacements for all trials
                    displacements = []
                    for trial in sim.trials[:traj_limit]:
                        tps = trial.get_timepoints(session).all()
                        pos_idx_start = np.array((tps[0].xidx, tps[0].yidx, tps[0].zidx))
                        if n_timesteps > len(tps) - 1:
                            # skip if the trajectory has ended by n_timesteps
                            continue

                        pos_idx_end = np.array((tps[n_timesteps].xidx,
                                                tps[n_timesteps].yidx,
                                                tps[n_timesteps].zidx))
                        displacements += [(pos_idx_end - pos_idx_start).astype(int)]

                    displacements = np.array(displacements)

                    # build the histogram
                    x_ub = min(n_timesteps + 1, sim.env.nx)
                    x_lb = -x_ub
                    y_ub = min(n_timesteps + 1, sim.env.ny)
                    y_lb = -y_ub
                    z_ub = min(n_timesteps + 1, sim.env.nz)
                    z_lb = -z_ub

                    x_bins = np.arange(x_lb, x_ub) + 0.5
                    y_bins = np.arange(y_lb, y_ub) + 0.5
                    z_bins = np.arange(z_lb, z_ub) + 0.5

                    displacement_histogram, _ = \
                        np.histogramdd(displacements, bins=(x_bins, y_bins, z_bins))

                    # create the data model and store it
                    displacement_hist_data_model = \
                        models.SimulationAnalysisDisplacementAfterNTimestepsHistogram()
                    displacement_hist_data_model.n_timesteps = n_timesteps
                    displacement_hist_data_model.simulation = sim
                    displacement_hist_data_model.shape = displacement_histogram.shape
                    displacement_hist_data_model. \
                        store_data(session, displacement_histogram.astype(int))
                    session.add(displacement_hist_data_model)

                    session.commit()


if __name__ == '__main__':
    main()