"""Generate infotaxis trials in simulated wind tunnel with simulated plume."""

SCRIPTID = 'generate_wind_tunnel_matched_trials'

import numpy as np

from insect import Insect
from plume import CollimatedPlume
from trial import Trial

from db_api import models
from db_api.connect import engine, session

from config.generate_trials import *

# get geom_config_group
geom_config_group = session.query(models.GeomConfigGroup).get(GEOMCONFIGGROUPID)

# create simulation
sim = models.Simulation(id=SIMULATIONID, description=SIMULATIONDESCRIPTION)
sim.env, sim.dt = ENV, DT
sim.heading_smoothing = HEADINGSMOOTHING
sim.geom_config_group = geom_config_group
session.add(sim)

# create and save plume
pl = CollimatedPlume(env=ENV, dt=DT)
pl.set_params(**PLUMEPARAMS)
pl.generate_orm(models, sim=sim)
session.add(pl.orm)

# create and save insect with known plume parameters
ins = Insect(env=ENV, dt=DT)
ins.set_params(**INSECTPARAMS)
ins.loglike_function = LOGLIKE
ins.generate_orm(models, sim=sim)
session.add(ins.orm)

# create ongoing run
ongoing_run = models.OngoingRun(trials_completed=0, simulations=[sim])
session.add(ongoing_run)

session.commit()

# generate trials
for tctr in range(NTRIALS):

    # randomly pick geom config
    geom_config = np.random.choice(geom_config_group.geom_configs)

    # set insect starting position
    ins.set_pos(geom_config.start_idx, is_idx=True)

    # initialize plume and insect and create trial
    pl.initialize()
    ins.initialize()

    trial = Trial(pl=pl, ins=ins, nsteps=geom_config.duration)

    # run trial, plotting along the way if necessary
    for step in xrange(geom_config.duration - 1):
        trial.step()

        if trial.at_src:
            print 'Found source after {} timesteps.'.format(trial.ts)
            break

    # save trial
    trial.add_timepoints(models, session=session, heading_smoothing=sim.heading_smoothing)
    trial.generate_orm(models)
    trial.orm.geom_config = geom_config
    trial.orm.simulation = sim
    session.add(trial.orm)

    # update ongoing_run
    ongoing_run.trials_completed = tctr + 1
    session.add(ongoing_run)

    # commit
    session.commit()