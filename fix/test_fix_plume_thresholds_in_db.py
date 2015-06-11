import unittest
import fix_plume_thresholds_in_db
from db_api.connect import session
from db_api import models

SIM_TEMPLATES = ('wind_tunnel_discretized_copies_fruitfly_0.3mps_checkerboard_floor_odor_{}',
                 'wind_tunnel_discretized_copies_fruitfly_0.4mps_checkerboard_floor_odor_{}',
                 'wind_tunnel_discretized_copies_fruitfly_0.6mps_checkerboard_floor_odor_{}')


class MainTestCase(unittest.TestCase):

    def setUp(self):
        fix_plume_thresholds_in_db.main()

    def test_all_odor_none_and_odor_afterodor_thresholds_correct_value(self):
        for sim_template in SIM_TEMPLATES:
            for odor_state in ('none', 'afterodor'):
                sim_id = sim_template.format(odor_state)
                sim = session.query(models.Simulation).get(sim_id)
                plume_params = {pp.name: pp.value for pp in sim.plume.plume_params}
                self.assertEqual(plume_params['threshold'], -1)

        for sim_template in SIM_TEMPLATES:
            sim_id = sim_template.format('on')
            sim = session.query(models.Simulation).get(sim_id)
            plume_params = {pp.name: pp.value for pp in sim.plume.plume_params}
            self.assertGreater(plume_params['threshold'], 0)


if __name__ == '__main__':
    unittest.main()