from __future__ import division

SCRIPT_ID = 'generate_discretized_wind_tunnel_trajectory_copies_and_geom_configs'
SCRIPT_NOTES = 'Load all wind tunnel trajectories and make discretized copies of them for direct comparison with infotaxis-begotten trajectories. This also requires making a geom_config_group and geom_configs for each trajectory.'

import os
import imp
import numpy as np

from db_api.connect import session
from db_api import models
from db_api import add_script_execution

from config.generate_discretized_wind_tunnel_trajectory_copies_and_geom_configs import *


# get wind tunnel connection and models
wt_models = imp.load_source('db_api.models', os.path.join(WT_REPO, 'db_api', 'models.py'))
wt_session = imp.load_source('db_api.connect', os.path.join(WT_REPO, 'db_api', 'connect.py')).session

# add script execution to database
add_script_execution(script_id=SCRIPT_ID, session=session, multi_use=True, notes=SCRIPT_NOTES)
session.commit()


for experiment_id in EXPERIMENT_IDS:

    for odor_state in ODOR_STATES:

        # make geom_config_group
        geom_config_group_id = '{}_{}_odor_{}'.format(GEOM_CONFIG_GROUP_ID, experiment_id, odor_state)
        geom_config_group = models.GeomConfigGroup(id=geom_config_group_id)

        # make simulation
        sim_id = '{}_{}_odor_{}'.format('wind_tunnel_copy_discretized', experiment_id, odor_state)
        sim = models.Simulation(id=sim_id)
        sim.env, sim.dt = ENV, DT
        sim.heading_smoothing = 0
        sim.geom_config_group = geom_config_group

        # make plume
        pl = CollimatedPlume(env=ENV, dt=DT)
        pl.set_params(**PLUME_PARAMS)
        if odor_state == 'on':
            pl.set_params(max_conc=488)
        else:
            pl.set_params(max_conc=0)
        pl.generate_data_model(models, sim=sim)
        sim.plume = pl.data_model
        session.add(sim)

        # loop over all wind tunnel trajectories
        trajs = wt_session.query(wt_models.Trajectory). \
            filter(Trajectory.experiment_id==experiment_id). \
            filter(Trajectory.odor_state==odor_state). \
            filter(Trajectory.clean==True)

        for traj in trajs:

            # create discretized version of trajectory
            trial = TrialFromTraj(timepoints, pl)

            # get geom_config, add extensions
            geom_config = mdoels.GeomConfig()
            geom_config.start_idx = trial.pos_idx[0]
            geom_config.duration = trial.ts + 1
            geom_config.geom_config_extension_real_trajectory = \
                models.GeomConfigExtensionRealTrajectory(real_trajectory_id=traj.id)

            # add timepoints to trial and generate data model
            trial.add_timepoints(models, session=session, heading_smoothing=sim.heading_smoothing)
            trial.generate_data_model(models)
            # bind simulation and geom_config to trial
            trial.data_model.sim = sim
            trial.data_model.geom_config = geom_config
            # add trial
            session.add(trial.data_model)

            # bind trial to geom_config's extension
            geom_config.geom_config_extension_real_trajectory.trial = trial

            geom_config_group.geom_configs += [geom_config]





