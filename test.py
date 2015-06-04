from __future__ import division
import unittest

import numpy as np

from plume import BasicPlume, CollimatedPlume, Environment3d
from insect import Insect
from trial import Trial, TrialFromTraj
from logprob_odor import binary_advec_diff_tavg


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


class DiscretizationTestCase(unittest.TestCase):

    def setUp(self):
        xrbins = np.linspace(0, 1., 11)
        yrbins = np.linspace(0, 1., 11)
        zrbins = np.linspace(0, 1., 11)
        self.env = Environment3d(xrbins, yrbins, zrbins)

        self.pl = CollimatedPlume(env=self.env, dt=.01)
        self.pl.set_params(20, 10, .5, .5, .5, .5)

    def test_straight_line_trajectory_discretization_by_trial_instance(self):

        # this trajectory should have 11 timesteps when mapped onto the grid in env
        x = 0.55 * np.ones((30,))
        y = np.linspace(.15, .75, 30)
        z = np.linspace(.45, .05, 30)
        positions = np.array([x, y, z]).T

        trial = TrialFromTraj(positions, self.pl)

        # check to make sure duration is correct
        self.assertEqual(trial.ts, 10)

    def test_perfectly_diagonal_trajectory_discretization_by_trial_instance(self):
        x = np.linspace(.15, .75, 40)
        y = np.linspace(.15, .75, 40)
        z = np.linspace(.15, .75, 40)
        positions = np.array([x, y, z]).T

        trial = TrialFromTraj(positions, self.pl)

        # check to make sure duration is correct
        true_duration = 19
        self.assertEqual(trial.ts + 1, true_duration)

        # check that each pos idx is one step away from the previous pos idx
        for pi_ctr in range(trial.ts):
            this_pos_idx = np.array(trial.pos_idx[pi_ctr])
            next_pos_idx = np.array(trial.pos_idx[pi_ctr + 1])
            self.assertEqual(np.abs(this_pos_idx - next_pos_idx).sum(), 1)

    def test_random_walk_trajectory_discretization_by_trial_instance(self):

        # loop over some more random trajectories
        for _ in range(5):

            x = 0.5 + np.random.normal(0, .003, (1000,)).sum()
            y = 0.5 + np.random.normal(0, .003, (1000,)).sum()
            z = 0.5 + np.random.normal(0, .003, (1000,)).sum()
            positions = np.array([x, y, z]).T
            # truncate positions if any of them go beyond 1 or 0
            outside_env = [ts for ts, pos in enumerate(positions) if np.any(pos > 1) or np.any(pos < 0)]
            if outside_env:
                positions = positions[:outside_env[0]]

            trial = TrialFromTraj(positions, self.pl)

            # check that each pos idx is one step away from the previous pos idx
            for pi_ctr in range(trial.ts):
                this_pos_idx = np.array(trial.pos_idx[pi_ctr])
                next_pos_idx = np.array(trial.pos_idx[pi_ctr + 1])
                self.assertEqual(np.abs(this_pos_idx - next_pos_idx).sum(), 1)

            # check that first and last pos idx are what env would give them
            first_pos_idx_env = np.array(self.env.idx_from_pos[positions[0]])
            last_pos_idx_env = np.array(self.env.idx_from_pos[positions[-1]])
            np.testing.assert_array_equal(first_pos_idx_env, np.array(trial.pos_idx[0]))
            np.testing.assert_array_equal(last_pos_idx_env, np.array(trial.pos_idx[trial.ts]))


class MathTestCase(unittest.TestCase):

    # make sure wind direction is correctly used in mathematics
    def test_wind_direction(self):

        xrbins = np.linspace(-0.3, 1.0, 66)
        yrbins = np.linspace(-0.15, 0.15, 16)
        zrbins = np.linspace(-0.15, 0.15, 16)
        env = Environment3d(xrbins, yrbins, zrbins)

        plume_params = {
            'w': 0.4,  # wind (m/s)
            'r': 500,  # source emission rate
            'd': 0.12,  # diffusivity (m^2/s)
            'a': .002,  # searcher size (m)
            'tau': 1000,  # particle lifetime (s)
        }

        pl = BasicPlume(env=env, dt=0.01)
        pl.set_params(**plume_params)
        pl.set_src_pos((0.3, 0.0, 0.0))
        pl.initialize()

        c0 = pl.conc[pl.env.idx_from_pos((0.4, 0.0, 0.0))]
        c1 = pl.conc[pl.env.idx_from_pos((0.2, 0.0, 0.0))]
        c2 = pl.conc[pl.env.idx_from_pos((0.2, 0.05, 0.0))]
        c3 = pl.conc[pl.env.idx_from_pos((0.4, 0.05, 0.0))]
        self.assertGreater(c0, c1)
        self.assertGreater(c1, c2)
        self.assertGreater(c0, c3)


        plume_params = {
            'w': 0.4,  # wind (m/s)
            'r': 0.00000001,  # source emission rate
            'd': 0.12,  # diffusivity (m^2/s)
            'a': .002,  # searcher size (m)
            'tau': 1000,  # particle lifetime (s)
        }

        pl = BasicPlume(env=env, dt=0.01)
        pl.set_params(**plume_params)
        pl.set_src_pos((0.3, 0.0, 0.0))
        pl.initialize()

        ins = Insect(env=env, dt=0.01)
        ins.set_params(**plume_params)
        ins.loglike_function = binary_advec_diff_tavg
        start_pos = (0.7, 0.0, 0.0)
        start_idx = ins.env.idx_from_pos(start_pos)
        ins.set_pos(start_pos)
        ins.initialize()

        tr = Trial(pl=pl, ins=ins)

        self.assertTrue(np.isinf(ins.logprob[start_idx]))
        self.assertLess(ins.logprob[start_idx], 0)

        uw_idx = list(start_idx)
        uw_idx[0] -= 3
        uw_idx = tuple(uw_idx)
        dw_idx = list(start_idx)
        dw_idx[0] += 3
        dw_idx = tuple(dw_idx)
        uw_prob = ins.logprob[uw_idx]
        dw_prob = ins.logprob[dw_idx]

        self.assertLess(uw_prob, dw_prob)

        uw_idx2 = list(uw_idx)
        uw_idx2[0] -= 1
        uw_idx2 = tuple(uw_idx2)
        uw_prob2 = ins.logprob[uw_idx2]

        self.assertLess(uw_prob, uw_prob2)

        dw_idx2 = list(dw_idx)
        dw_idx2[0] += 1
        dw_idx2 = tuple(dw_idx2)
        dw_prob2 = ins.logprob[dw_idx2]

        self.assertLess(dw_prob, dw_prob2)


if __name__ == '__main__':
    unittest.main()