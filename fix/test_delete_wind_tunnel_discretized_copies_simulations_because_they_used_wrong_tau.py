SIM_TEMPLATE = 'wind_tunnel_discretized_copies%'

import unittest

import delete_wind_tunnel_discretized_copies_simulations_because_they_used_wrong_tau

from db_api import models
from db_api.connect import session


class MainTestCase(unittest.TestCase):

    def test_no_sims_anymore(self):
        sims = session.query(models.Simulation).filter(models.Simulation.id.like(SIM_TEMPLATE))
        self.assertEqual(len(sims.all()), 0)


if __name__ == '__main__':
    delete_wind_tunnel_discretized_copies_simulations_because_they_used_wrong_tau.main()
    unittest.main()