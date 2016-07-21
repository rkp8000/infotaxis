"""
THIS CODE IS NO LONGER IN USE, SINCE IT HAS BEEN ABSORBED BY "make_wind_tunnel_discretized...".
IT WAS USED ONCE BEFORE THAT CODE HAD BEEN UPDATED.
"""

from __future__ import division

SCRIPTID = 'extend_wind_tunnel_matched_geom_configs_with_trajectory_ids'
SCRIPTNOTES = 'Go through all geom_configs grabbed from real wind tunnel trajectories and add to the extension table ' \
              'the trajectory ids they correspond to.'

import imp

from db_api.connect import session
from db_api.models import GeomConfigGroup, GeomConfigExtensionRealTrajectory
from db_api import add_script_execution

from config.extend_geom_configs_with_trajectory_ids import *

# get wind tunnel connection and models
wt_connect = imp.load_source('db_api.connect', '/Users/rkp/Dropbox/Repositories/wind_tunnel/db_api/connect.py')
wt_models = imp.load_source('db_api.models', '/Users/rkp/Dropbox/Repositories/wind_tunnel/db_api/models.py')
wt_session = wt_connect.session

# add script execution to database
add_script_execution(script_id=SCRIPTID, session=session, multi_use=False, notes=SCRIPTNOTES)
session.commit()

for EXPERIMENTID, GEOMCONFIGGROUPID in zip(EXPERIMENTIDS, GEOMCONFIGGROUPIDS):
    try:
        # loop over all odor states
        for odor_state in ODORSTATES:
            geom_config_group_id = GEOMCONFIGGROUPID + '_odor_' + odor_state
            geom_config_group = session.query(GeomConfigGroup).get(geom_config_group_id)

            print 'current geom_config_group: {}'.format(geom_config_group_id)

            # get all starting positions and durations from real data
            expt = wt_session.query(wt_models.Experiment).get(EXPERIMENTID)
            traj_configs = []

            for tctr, traj in enumerate(expt.trajectories):
                if traj.clean and traj.odor_state == odor_state:
                    start_tp = wt_session.query(wt_models.Timepoint).get(traj.start_timepoint_id)
                    start_idx = ENV.idx_from_pos((start_tp.x, start_tp.y, start_tp.z))

                    duration = int(np.ceil(traj.duration * .01 / DT))
                    traj_id = traj.id
                    traj_configs += [(start_idx, duration, traj_id)]

            # check to make sure geom configs from all starting positions match
            for tctr, traj_config in enumerate(traj_configs):
                start_idx, duration, traj_id = traj_config
                start_xidx, start_yidx, start_zidx = start_idx

                # make sure these all line up with what's there already
                geom_config = geom_config_group.geom_configs[tctr]

                if geom_config.start_idx == start_idx and geom_config.duration == duration:
                    # add trajectory id to this geom_config
                    geom_config.geom_config_extension_real_trajectory = \
                        GeomConfigExtensionRealTrajectory(real_trajectory_id=traj_id)
                else:
                    print 'error!'

            session.add(geom_config_group)
    except Exception, e:
        session.rollback()
        print e
    else:
        session.commit()