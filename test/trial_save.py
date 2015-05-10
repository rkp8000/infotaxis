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

from db_api import mappings
from db_api.connect import engine, session, TESTCXN

from config.trial_save import *


if not TESTCXN:
    raise ValueError('TESTCXN is not set to True. Aborting test...')

mappings.Base.metadata.create_all(engine)

# create simulation
sim = mappings.Simulation()
sim.id, sim.description, sim.total_trials = ID, DESCRIPTION, TOTALTRIALS
sim.env, sim.dt = ENV, DT
session.add(sim)

# get geom_config_group and add simulation to it
geom_config_group = session.query(mappings.GeomConfigGroup).get(GEOMCONFIGGROUP)
geom_config_group.simulations += [sim]
session.add(geom_config_group)

# create and save plume
pl = BasicPlume(env=ENV, dt=DT)
pl.set_params(**PLUME_PARAMS)
pl.set_orm(mappings, sim=sim)
session.add(pl.orm)

# create and save insect
ins = Insect(env=ENV, dt=DT)
ins.set_params(**PLUME_PARAMS)
ins.loglike_function = LOGLIKE
ins.set_orm(mappings, sim=sim)
session.add(ins.orm)

# create ongoing run
ongoing_run = mappings.OngoingRun(trials_completed=0, simulations=[sim])
session.add(ongoing_run)

session.commit()

# loop over all trials
for tctr in range(TOTALTRIALS):

    # pick random configuration
    geom_config = np.random.choice(geom_config_group.geom_configs)

    # set source position
    src_pos_idx = geom_config.src_xidx, geom_config.src_yidx, geom_config.src_zidx
    pl.set_src_pos(src_pos_idx, is_idx=True)

    # set insect starting position
    start_pos_idx = geom_config.start_xidx, geom_config.start_yidx, geom_config.start_zidx
    ins.set_pos(start_pos_idx, is_idx=True)

    # create trial
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
    # trial.save_timepoints(mappings, session=session)
    # trial.set_orm(mappings)
    # session.add(trial.orm)

    start_tp_id, end_tp_id = None, None
    # add timepoints
    for tp_ctr in xrange(trial.ts + 1):
        tp = mappings.Timepoint()
        tp.xidx, tp.yidx, tp.zidx = trial.pos_idx[tp_ctr]
        session.add(tp)

        # get timepoint start and end ids if first iteration
        if tp_ctr == 0:
            session.flush()
            start_tp_id = tp.id
            end_tp_id = start_tp_id + trial.ts

    # create trial
    tr_rel = mappings.Trial()
    tr_rel.start_timepoint_id = start_tp_id
    tr_rel.end_timepoint_id = end_tp_id

    # add trial info to trial
    tr_info = mappings.TrialInfo()
    tr_info.duration = trial.ts + 1
    tr_info.found_src = trial.at_src
    tr_rel.trial_info = [tr_info]

    # add trial to geom_config and simulation
    geom_config.trials += [tr_rel]
    sim.trials += [tr_rel]

    session.add(geom_config)
    session.add(sim)

    # update ongoing_run
    ongoing_run.trials_completed = tctr + 1
    session.add(ongoing_run)

    # commit
    session.commit()