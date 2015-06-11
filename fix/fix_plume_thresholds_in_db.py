from __future__ import print_function, division

SCRIPT_ID = 'fix_collimated_plume_thresholds_of_odor_none_and_afterodor_in_db'
SCRIPT_NOTES = 'Change thresholds of plumes corresponding to all wind_tunnel_discretized_copies simulations to -1 in database, since these weren\'t actually stored correctly.'

SIM_TEMPLATES = ('wind_tunnel_discretized_copies_fruitfly_0.3mps_checkerboard_floor_odor_{}',
                 'wind_tunnel_discretized_copies_fruitfly_0.4mps_checkerboard_floor_odor_{}',
                 'wind_tunnel_discretized_copies_fruitfly_0.6mps_checkerboard_floor_odor_{}')

ODOR_STATES = ('none', 'afterodor')

from db_api.connect import session
from db_api import models, add_script_execution


def main():
    add_script_execution(SCRIPT_ID, session=session, notes=SCRIPT_NOTES, multi_use=False)

    for sim_template in SIM_TEMPLATES:
        for odor_state in ODOR_STATES:

            sim_id = sim_template.format(odor_state)

            sim = session.query(models.Simulation).get(sim_id)

            threshold_param = session.query(models.PlumeParam). \
                filter(models.PlumeParam.plume==sim.plume). \
                filter(models.PlumeParam.name=='threshold').first()

            threshold_param.value = -1

            session.add(threshold_param)

    session.commit()


if __name__ == '__main__':
    main()