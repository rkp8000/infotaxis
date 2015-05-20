"""
Generate several 3D infotaxis trials using a basic plume based on the
advection-diffusion equation with turbulent diffusivity.

Geometry is set as wind tunnel geometry and R is varied.
"""

import numpy as np

from insect import Insect
from plume import BasicPlume
from trial import Trial

from db_api import models
from db_api.connect import engine, session

from config.generate_trials import *


models.Base.metadata.create_all(engine)

# loop over all Rs (all simulations)
for r in Rs:

    # create simulation
    sim = models.Simulation()
    sim.id, sim.description = ID.format(r), DESCRIPTION
    sim.env, sim.dt = ENV, DT
    sim.heading_smoothing = HEADING_SMOOTHING
    # bind geom_config_group
    sim.geom_config_group = session.query(models.GeomConfigGroup).get(GEOMCONFIGGROUP)
    sim.total_trials = len(sim.geom_config_group.geom_configs)
    session.add(sim)

    # create and save plume
    PLUME_PARAMS['r'] = r
    pl = BasicPlume(env=ENV, dt=DT)
    pl.set_params(**PLUME_PARAMS)
    pl.generate_orm(models, sim=sim)
    session.add(pl.orm)

    # create and save insect with known plume parameters
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
    for tctr, geom_config in enumerate(sim.geom_config_group.geom_configs):

        # set source position
        pl.set_src_pos(geom_config.src_idx, is_idx=True)

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