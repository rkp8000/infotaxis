from __future__ import division
import unittest

import numpy as np

from plume import BasicPlume, Environment3d
from insect import Insect
from trial import Trial
from logprob_odor import binary_advec_diff_tavg


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


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