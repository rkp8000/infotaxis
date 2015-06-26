from __future__ import print_function, division

TRAJ_LIMIT = 5
SIM_ID_0 = 'wind_tunnel_discretized_matched_r1000_d0.12_fruitfly_0.4mps_checkerboard_floor_odor_on'
SIM_ID_1 = 'wind_tunnel_discretized_copies_fruitfly_0.3mps_checkerboard_floor_odor_afterodor'
N_N_TIMESTEPS = 6

import unittest
import matplotlib.pyplot as plt

import make_displacement_after_n_timesteps_histograms_3d

from db_api import models
from db_api.connect import session


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


class MainTestCase(unittest.TestCase):

    def test_correct_number_of_histograms_made(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)

            self.assertEqual(N_N_TIMESTEPS,
                             len(sim.analysis_displacement_after_n_timesteps_histograms))

    def test_histograms_are_correct_size_and_have_correct_number_of_points(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)

            hists = sim.analysis_displacement_after_n_timesteps_histograms
            [hist.fetch_data(session) for hist in hists]

            for h_ctr, hist in enumerate(hists):
                n_timesteps = hist.n_timesteps
                nx = min(2 * sim.env.nx - 1, 2 * n_timesteps + 1)
                ny = min(2 * sim.env.ny - 1, 2 * n_timesteps + 1)
                nz = min(2 * sim.env.nz - 1, 2 * n_timesteps + 1)

                self.assertEqual(hist._data.shape, (nx, ny, nz))

                if n_timesteps == 1:
                    self.assertEqual(hist._data[1, 1, 1], 0)
                    print(hist._data.sum())

                if h_ctr < len(hists) - 1:
                    self.assertGreaterEqual(hist._data.sum(), hists[h_ctr + 1]._data.sum())

    def test_plotting_of_histograms(self):
        for sim_id in [SIM_ID_0, SIM_ID_1]:
            sim = session.query(models.Simulation).get(sim_id)

            hists = sim.analysis_displacement_after_n_timesteps_histograms
            [hist.fetch_data(session) for hist in hists]

            fig, axs = plt.subplots(3, 2)
            axs[0, 0].matshow(hists[2].xy.T, origin='lower')
            axs[1, 0].matshow(hists[2].xz.T, origin='lower')
            axs[2, 0].matshow(hists[2].yz.T, origin='lower')

            axs[0, 1].matshow(hists[4].xy.T, origin='lower')
            axs[1, 1].matshow(hists[4].xz.T, origin='lower')
            axs[2, 1].matshow(hists[4].yz.T, origin='lower')

            plt.show(block=True)

if __name__ == '__main__':
    make_displacement_after_n_timesteps_histograms_3d.main(TRAJ_LIMIT)
    unittest.main()