from __future__ import division
import unittest

import numpy as np
import matplotlib.pyplot as plt

from plume import BasicPlume, EmptyPlume, Environment3d
from insect import Insect
from trial import Trial, TrialFromPositionSequence
from logprob_odor import binary_advec_diff_tavg


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


class TrialFromPositionSequenceTestCase(unittest.TestCase):

    def setUp(self):
        xrbins = np.linspace(-0.3, 1.0, 66)
        yrbins = np.linspace(-0.15, 0.15, 16)
        zrbins = np.linspace(-0.15, 0.15, 16)
        self.env = Environment3d(xrbins, yrbins, zrbins)

        self.pl = EmptyPlume(self.env)
        self.pl.initialize()
        self.ins = Insect(env=self.env, dt=.01)
        self.ins.set_params(w=0.4, r=1000, d=0.12, a=0.002, tau=1000)
        self.ins.loglike_function = binary_advec_diff_tavg

    def test_source_probability_updating(self):
        xs = np.linspace(-.1, .7, 100)
        ys = np.zeros(xs.shape, dtype=float)
        zs = .06 * np.ones(xs.shape, dtype=float)
        positions = np.array([xs, ys, zs]).T

        self.ins.initialize()
        trial = TrialFromPositionSequence(positions, self.pl, self.ins)

        # make sure insect has gone strictly downwind in a straight line
        self.assertEqual(trial.pos_idx[0, 1], trial.pos_idx[-1, 1])
        self.assertEqual(trial.pos_idx[0, 2], trial.pos_idx[-1, 2])
        self.assertLess(trial.pos_idx[0, 0], trial.pos_idx[-1, 0])

        for pctr, pos_idx in enumerate(trial.pos_idx[:-1]):

            # make sure log source probability is -inf at all visited locations
            self.assertLess(trial.ins.logprob[tuple(pos_idx)], 0)
            self.assertTrue(np.isinf(trial.ins.logprob[tuple(pos_idx)]))

            # make sure entropy is greater than zero for all timepoints
            self.assertGreater(trial.entropies[pctr], 0)

            # make sure entropy is lower than previous entropy
            if pctr >= 1:
                self.assertLess(trial.entropies[pctr], trial.entropies[pctr - 1])

        # make sure source probability is lower upwind than downwind of insect
        cur_pos_idx = trial.pos_idx[-1]
        for displacement in range(1, 5):
            uw_pos_idx = np.array(cur_pos_idx) - np.array([displacement, 1, 0])
            dw_pos_idx = np.array(cur_pos_idx) + np.array([displacement, 1, 0])
            self.assertLess(trial.ins.logprob[tuple(uw_pos_idx)],
                            trial.ins.logprob[tuple(dw_pos_idx)])

        plt.matshow(trial.ins.logprobxy.T, origin='lower')
        plt.show(block=True)


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

        self.assertTrue(np.isinf(tr.ins.logprob[start_idx]))
        self.assertLess(tr.ins.logprob[start_idx], 0)

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