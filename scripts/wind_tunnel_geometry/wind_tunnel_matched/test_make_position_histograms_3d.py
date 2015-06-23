from __future__ import print_function, division

TRAJ_LIMIT = 8
SIM_ID_0 = 'wind_tunnel_discretized_matched_r1000_d0.12_fruitfly_0.4mps_checkerboard_floor_odor_on'
SIM_ID_1 = 'wind_tunnel_discretized_copies_fruitfly_0.3mps_checkerboard_floor_odor_afterodor'


import unittest
import matplotlib.pyplot as plt

import make_position_histograms_3d

from db_api import models
from db_api.connect import session


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


class MainTestCase(unittest.TestCase):

    def test_correct_simulations_analyzed(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)
            self.assertEqual(sim.analysis_position_heatmap.simulation, sim)

    def test_histogram_dimensions_correct(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)
            sim.analysis_position_heatmap.fetch_data(session)
            heatmap_xy = sim.analysis_position_heatmap.xy
            self.assertEqual(heatmap_xy.shape[0], sim.env.nx)
            self.assertEqual(heatmap_xy.shape[1], sim.env.ny)

            heatmap_xz = sim.analysis_position_heatmap.xz
            self.assertEqual(heatmap_xz.shape[0], sim.env.nx)
            self.assertEqual(heatmap_xz.shape[1], sim.env.nz)

            heatmap_yz = sim.analysis_position_heatmap.yz
            self.assertEqual(heatmap_yz.shape[0], sim.env.ny)
            self.assertEqual(heatmap_yz.shape[1], sim.env.nz)

    def test_plot_heatmaps(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)
            sim.analysis_position_heatmap.fetch_data(session)
            heatmap_xy = sim.analysis_position_heatmap.xy
            heatmap_xz = sim.analysis_position_heatmap.xz
            heatmap_yz = sim.analysis_position_heatmap.yz

            fig, axs = plt.subplots(1, 3)
            axs[0].matshow(heatmap_xy.T, origin='lower', extent=sim.env.extentxy)
            axs[1].matshow(heatmap_xz.T, origin='lower', extent=sim.env.extentxz)
            axs[2].matshow(heatmap_yz.T, origin='lower', extent=sim.env.extentyz)

            plt.show()


if __name__ == '__main__':
    make_position_histograms_3d.main(TRAJ_LIMIT)
    unittest.main()