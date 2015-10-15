"""
Generate one infotaxis trial for every real trial, matching starting location, number of timesteps in trajectory, and plume geometry.
"""
from __future__ import print_function, division

SCRIPTID = 'generate_wind_tunnel_discretized_matched_trials_one_for_one'
SCRIPTNOTES = 'Run for all experiments and odor states with d = 0.02 m^2/s.'


from insect import Insect
from plume import CollimatedPlume, SpreadingGaussianPlume
from trial import Trial

from db_api import models
from db_api.connect import session
from db_api import add_script_execution

from config.generate_wind_tunnel_discretized_matched_trials_one_for_one import *


def main(traj_limit=None):
    # add script execution to database
    add_script_execution(SCRIPTID, session=session, multi_use=True, notes=SCRIPTNOTES)

    for expt in EXPERIMENTS:
        if '0.3mps' in expt:
            w = 0.3
        elif '0.4mps' in expt:
            w = 0.4
        elif '0.6mps' in expt:
            w = 0.6

        insect_params = INSECT_PARAMS.copy()
        insect_params['w'] = w

        for odor_state in ODOR_STATES:

            print('Running simulation for expt "{}" with odor "{}"...'.
                  format(expt, odor_state))

            # get geom_config_group for this experiment and odor state
            geom_config_group_id = GEOM_CONFIG_GROUP_ID.format(expt, odor_state)
            geom_config_group = session.query(models.GeomConfigGroup).get(geom_config_group_id)

            # get wind tunnel copy simulation so we can match plume and insect
            # note we select the first simulation that is of this type and corresponds to the
            # right geom_config_group, since we only use the plume from it, which is independent
            # of what insect parameters were used
            #
            # for instance, the plume bound to a simulation in which the insect had D = 0.6 and that
            # bound to a simulation where D = 0.4 will be the same, since it is only the insect's
            # internal model that has changed
            wt_copy_sims = session.query(models.Simulation).\
                filter(models.Simulation.geom_config_group == geom_config_group).\
                filter(models.Simulation.id.like(WIND_TUNNEL_DISCRETIZED_SIMULATION_ID_PATTERN))

            # get plume from corresponding discretized real wind tunnel trajectory
            if 'fruitfly' in expt:
                pl = CollimatedPlume(env=ENV, dt=-1, orm=wt_copy_sims.first().plume)
            elif 'mosquito' in expt:
                pl = SpreadingGaussianPlume(env=ENV, dt=-1, orm=wt_copy_sims.first().plume)

            # create insect
            # note: we will actually make a new insect for each trial, since the dt's vary;
            # here we just set dt=-1, since this doesn't get stored in the db anyhow
            ins = Insect(env=ENV, dt=-1)
            ins.set_params(**insect_params)
            ins.generate_orm(models)

            # create simulation
            sim_id = SIMULATION_ID.format(insect_params['r'],
                                          insect_params['d'],
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
                ins.set_params(**insect_params)
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
                ongoing_run.trials_completed = gctr + 1
                session.add(ongoing_run)

                session.commit()

if __name__ == '__main__':
    main()