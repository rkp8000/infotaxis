"""
Generate one infotaxis trial for every real trial, matching starting location, number of timesteps in trajectory, and plume geometry.
"""
from __future__ import print_function, division

SCRIPTID = 'generate_wind_tunnel_matched_trials_one_for_one'
SCRIPTNOTES = 'Run for all experiments and all odor states.'


from insect import Insect
from plume import CollimatedPlume
from trial import Trial

from db_api import models
from db_api.connect import session
from db_api import add_script_execution

from config.generate_trials_one_for_one import *


def main(traj_limit=None):
    # add script execution to database
    add_script_execution(SCRIPTID, session=session, multi_use=False, notes=SCRIPTNOTES)

    for expt in EXPERIMENTS:
        for odor_state in ODOR_STATES:

            print('Running simulation for expt "{}" with odor "{}"...'.
                  format(expt, odor_state))

            # get geom_config_group for this experiment and odor state
            geom_config_group_id = GEOM_CONFIG_GROUP_ID.format(expt, odor_state)
            geom_config_group = session.query(models.GeomConfigGroup).get(geom_config_group_id)

            # get wind tunnel copy simulation so we can match plume and insect
            wt_copy_sims = session.query(models.Simulation).\
                filter(models.Simulation.geom_config_group==geom_config_group).\
                filter(models.Simulation.id.like(WIND_TUNNEL_DISCRETIZED_SIMULATION_ID_PATTERN))

            # get plume from corresponding discretized real wind tunnel trajectory
            pl = CollimatedPlume(env=ENV, dt=-1, orm=wt_copy_sims.first().plume)


            # create insect
            # note: we will actually make a new insect for each trial, since the dt's vary;
            # here we just set dt=-1, since this doesn't get stored in the db anyhow
            ins = Insect(env=ENV, dt=-1)
            ins.set_params(**INSECT_PARAMS)
            ins.generate_orm(models)


            # create simulation
            sim_id = SIMULATION_ID.format(INSECT_PARAMS['r'],
                                          INSECT_PARAMS['d'],
                                          expt, odor_state)
            sim_desc = SIMULATION_DESCRIPTION.format(expt, odor_state)

            sim = models.Simulation(id=sim_id, description=sim_desc)
            sim.env = ENV
            sim.dt = -1
            sim.total_trials = len(geom_config_group.geom_configs)
            sim.heading_smoothing = 0
            sim.geom_config_group = geom_config_group

            sim.plume = pl.orm
            sim.insect = ins.orm

            session.add(sim)


            # create ongoing run
            ongoing_run = models.OngoingRun(trials_completed=0, simulations=[sim])
            session.add(ongoing_run)

            session.commit()

            # generate trials
            for gctr, geom_config in enumerate(geom_config_group.geom_configs):

                if gctr == traj_limit:
                    break

                # make new plume and insect with proper dts
                ins = Insect(env=ENV, dt=geom_config.extension_real_trajectory.avg_dt)
                ins.set_params(**INSECT_PARAMS)
                ins.loglike_function = LOGLIKE

                # set insect starting position
                ins.set_pos(geom_config.start_idx, is_idx=True)

                # initialize plume and insect and create trial
                pl.initialize()
                ins.initialize()

                trial = Trial(pl=pl, ins=ins, nsteps=geom_config.duration)

                # run trial
                for step in xrange(geom_config.duration - 1):
                    trial.step()

                # save trial
                trial.add_timepoints(models, session=session, heading_smoothing=sim.heading_smoothing)
                trial.generate_orm(models)
                trial.orm.geom_config = geom_config
                trial.orm.simulation = sim
                session.add(trial.orm)

                # update ongoing_run
                ongoing_run.trials_completed = gctr
                session.add(ongoing_run)

                session.commit()

if __name__ == '__main__':
    main()