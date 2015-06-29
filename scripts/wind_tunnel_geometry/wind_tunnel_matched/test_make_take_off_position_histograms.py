import unittest

SIM_ID_START = 'wind_tunnel_discretized_copies_'

import make_take_off_position_histograms

from db_api import models
from db_api.connect import session


class MainTestCase(unittest.TestCase):

    def test_correct_number_of_histograms_and_correct_size_and_count(self):

        hists = session.query(models.SimulationAnalysisTakeOffPositionHistogram). \
            filter(models.SimulationAnalysisTakeOffPositionHistogram.simulation_id. \
                   like(SIM_ID_START + '%'))

        self.assertEqual(len(hists.all()), 9)

        for hist in hists:
            hist.fetch_data(session)
            sim = hist.simulation
            shape = (sim.env.nx, sim.env.ny, sim.env.nz)
            self.assertEqual(shape, hist._data.shape)

            self.assertEqual(len(sim.trials), hist._data.sum())

            print(hist._data.sum())


if __name__ == '__main__':
    make_take_off_position_histograms.main()
    unittest.main()