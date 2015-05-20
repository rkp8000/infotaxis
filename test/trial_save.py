"""
Generate several 3D infotaxis trajectory as the insect flies through a basic plume.

Save these in the database using the SQLAlchemy DBAPI.

The settings and parameters of this demo are located in config/trial_save.py.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from insect import Insect
from plume import BasicPlume
from trial import Trial

from db_api import models
from db_api.connect import engine, session, TESTCXN

from config.trial_save import *


if not TESTCXN:
    raise ValueError('TESTCXN is not set to True. Aborting test...')

models.Base.metadata.create_all(engine)

# get geom_config_group
geom_config_group = session.query(models.GeomConfigGroup).get(GEOMCONFIGGROUP)

# create simulation
sim = models.Simulation()
sim.id, sim.description, sim.total_trials = ID, DESCRIPTION, TOTALTRIALS
sim.env, sim.dt = ENV, DT
sim.heading_smoothing = HEADING_SMOOTHING
sim.geom_config_group = geom_config_group
session.add(sim)

# create and save plume
pl = BasicPlume(env=ENV, dt=DT)
pl.set_params(**PLUME_PARAMS)
pl.generate_orm(models, sim=sim)
session.add(pl.orm)

# create and save insect
ins = Insect(env=ENV, dt=DT)
ins.set_params(**PLUME_PARAMS)
ins.loglike_function = LOGLIKE
ins.generate_orm(models, sim=sim)
session.add(ins.orm)

# create ongoing run
ongoing_run = models.OngoingRun(trials_completed=0, simulations=[sim])
session.add(ongoing_run)

session.commit()

# loop over all trials
for tctr in range(TOTALTRIALS):

    # pick random configuration
    geom_config = np.random.choice(geom_config_group.geom_configs)

    # set source position
    pl.set_src_pos(geom_config.src_idx, is_idx=True)

    # set insect starting position
    ins.set_pos(geom_config.start_idx, is_idx=True)

    # initialize plume and insect and create trial
    pl.initialize()
    ins.initialize()

    trial = Trial(pl=pl, ins=ins, nsteps=geom_config.duration)

    # run trial
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