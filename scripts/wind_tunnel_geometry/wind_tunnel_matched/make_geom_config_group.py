from __future__ import division
import imp
import numpy as np

from db_api.connect import session
from db_api.models import GeomConfig, GeomConfigGroup
from db_api import add_script_execution

from config.make_geom_config_group import *

# get wind tunnel connection and models
wt_connect = imp.load_source('db_api.connect', '/Users/rkp/Dropbox/Repositories/wind_tunnel/db_api/connect.py')
wt_models = imp.load_source('db_api.models', '/Users/rkp/Dropbox/Repositories/wind_tunnel/db_api/models.py')
wt_session = wt_connect.session

# add script execution to database
add_script_execution(script_id=SCRIPTID, session=session, multi_use=True)

try:
    # loop over all odor states
    for odor_state in ODORSTATES:
        geom_config_group_id = GEOMCONFIGGROUPID + '_odor_' + odor_state
        geom_config_group = GeomConfigGroup(id=geom_config_group_id)
        geom_config_group.description = GEOMCONFIGGROUPDESCRIPTION.format(EXPERIMENTID, odor_state)

        print 'current geom_config_group: {}'.format(geom_config_group_id)

        # get all starting positions and durations
        expt = wt_session.query(wt_models.Experiment).get(EXPERIMENTID)
        traj_configs = []

        for traj in expt.trajectories:
            if traj.clean and traj.odor_state == odor_state:
                start_tp = wt_session.query(wt_models.Timepoint).get(traj.start_timepoint_id)
                start_idx = ENV.idx_from_pos((start_tp.x, start_tp.y, start_tp.z))

                duration = int(np.ceil(traj.duration * .01 / DT))
                traj_configs += [(start_idx, duration)]

        # create geom configs from all starting positions
        for start_idx, duration in traj_configs:
            start_xidx, start_yidx, start_zidx = start_idx
            geom_config = GeomConfig(start_xidx=start_xidx, start_yidx=start_yidx,
                                     start_zidx=start_zidx, duration=duration)
            geom_config_group.geom_configs += [geom_config]

        session.add(geom_config_group)
except Exception, e:
    session.rollback()
    print e
else:
    session.commit()