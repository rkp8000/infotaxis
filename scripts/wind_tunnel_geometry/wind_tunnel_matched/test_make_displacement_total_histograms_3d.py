from __future__ import print_function, division

TRAJ_LIMIT = 5
SIM_ID_0 = 'wind_tunnel_discretized_matched_r1000_d0.12_fruitfly_0.4mps_checkerboard_floor_odor_on'
SIM_ID_1 = 'wind_tunnel_discretized_copies_fruitfly_0.3mps_checkerboard_floor_odor_afterodor'

import unittest
import matplotlib.pyplot as plt

import make_displacement_total_histograms_3d

from db_api import models
from db_api.connect import session


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


class MainTestCase(unittest.TestCase):

    def test_correct_simulations_analyzed(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)
            self.assertEqual(sim.analysis_displacement_total_histogram.simulation, sim)

    def test_histogram_dimensions_correct(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)
            sim.analysis_displacement_total_histogram.fetch_data(session)
            heatmap_xy = sim.analysis_displacement_total_histogram.xy
            self.assertEqual(heatmap_xy.shape[0], 2 * sim.env.nx - 1)
            self.assertEqual(heatmap_xy.shape[1], 2 * sim.env.ny - 1)

            heatmap_xz = sim.analysis_displacement_total_histogram.xz
            self.assertEqual(heatmap_xz.shape[0], 2 * sim.env.nx - 1)
            self.assertEqual(heatmap_xz.shape[1], 2 * sim.env.nz - 1)

            heatmap_yz = sim.analysis_displacement_total_histogram.yz
            self.assertEqual(heatmap_yz.shape[0], 2 * sim.env.ny - 1)
            self.assertEqual(heatmap_yz.shape[1], 2 * sim.env.nz - 1)

    def test_number_of_points_in_histogram_is_number_of_trials(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)
            sim.analysis_displacement_total_histogram.fetch_data(session)
            n_trials = len(sim.trials)
            n_points = sim.analysis_displacement_total_histogram.xy.sum()
            self.assertEqual(n_points, n_trials)
            n_points = sim.analysis_displacement_total_histogram.xz.sum()
            self.assertEqual(n_points, n_trials)
            n_points = sim.analysis_displacement_total_histogram.yz.sum()
            self.assertEqual(n_points, n_trials)

    def test_plot_heatmaps(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)
            sim.analysis_displacement_total_histogram.fetch_data(session)
            heatmap_xy = sim.analysis_displacement_total_histogram.xy
            heatmap_xz = sim.analysis_displacement_total_histogram.xz
            heatmap_yz = sim.analysis_displacement_total_histogram.yz

            extent_xy = sim.analysis_displacement_total_histogram.extent_xy
            extent_xz = sim.analysis_displacement_total_histogram.extent_xz
            extent_yz = sim.analysis_displacement_total_histogram.extent_yz

            fig, axs = plt.subplots(1, 3)
            axs[0].matshow(heatmap_xy.T, origin='lower', extent=extent_xy)
            axs[1].matshow(heatmap_xz.T, origin='lower', extent=extent_xz)
            axs[2].matshow(heatmap_yz.T, origin='lower', extent=extent_yz)

            plt.show()


if __name__ == '__main__':
    make_displacement_total_histograms_3d.main(TRAJ_LIMIT)
    unittest.main()