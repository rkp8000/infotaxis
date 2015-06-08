from __future__ import print_function, division

TRAJ_LIMIT = 10

import unittest
import make_wind_tunnel_discretized_geom_configs

from db_api.connect import session
from db_api import models


class DatabaseTestCase(unittest.TestCase):

    def setUp(self):
        try:
            make_wind_tunnel_discretized_geom_configs.main(TRAJ_LIMIT)
        except RuntimeError, e:
            print('Error: {}'.format(e))
            pass

    def test_correct_number_of_geom_configs_added(self):
        gcgs = session.query(models.GeomConfigGroup). \
            filter(models.GeomConfigGroup.id.like('wind_tunnel_matched_discretized%'))

        self.assertEqual(len(gcgs.all()), 9)

        for gcg in gcgs:
            self.assertEqual(len(gcg.geom_configs), TRAJ_LIMIT + 1)

            # make sure all geom_configs have geom_config_extension with all fields filled out
            for gc in gcg.geom_configs:
                self.assertGreater(gc.geom_config_extension_real_trajectory.avg_dt, 0)
                self.assertGreater(len(gc.geom_config_extension_real_trajectory.real_trajectory_id), 0)

if __name__ == '__main__':
    unittest.main()