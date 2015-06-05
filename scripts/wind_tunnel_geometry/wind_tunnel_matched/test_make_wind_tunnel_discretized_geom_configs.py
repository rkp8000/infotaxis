TRAJ_LIMIT = 10

import unittest
import make_wind_tunnel_discretized_geom_configs

from db_api.connect import session
from db_api import models


class DatabaseTest(unittest.TestCase):

    def setUp(self):
        make_wind_tunnel_discretized_geom_configs.main(TRAJ_LIMIT)

    def test_correct_number_of_geom_configs_added(self):
        gcgs = session.query(models.GeomConfigGroup). \
            filter(models.GeomConfigGroup.id.like('wind_tunnel_matched_discretized'))

        self.assertEqual(len(gcgs), 9)

        for gcg in gcgs:
            self.assertEqual(len(gcg.geom_configs), TRAJ_LIMIT)


if __name__ == '__main__':
    unittest.main()