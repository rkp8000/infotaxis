from __future__ import print_function, division

TRAJ_LIMIT = 8

import unittest
import generate_wind_tunnel_discretized_matched_trials_one_for_one

from db_api import models
from db_api.connect import session


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


class MainTestCase(unittest.TestCase):

    def setUp(self):
        try:
            generate_wind_tunnel_discretized_matched_trials_one_for_one.main(TRAJ_LIMIT)
        except Exception, e:
            pass

    def test_correct_number_of_simulations_trials_and_timepoints(self):
        sims = session.query(models.Simulation).\
            filter(models.Simulation.id.like('%wind_tunnel_discretized_matched%')).all()

        self.assertEqual(len(sims), 9)

        for sim in sims:
            self.assertEqual(len(sim.trials), TRAJ_LIMIT)

            for trial in sim.trials:
                self.assertEqual(len(trial.get_timepoints(session).all()), trial.trial_info.duration)
                self.assertEqual(len(trial.get_timepoints(session).all()), trial.geom_config.duration)

    def test_windspeeds_are_correct(self):
        sims = session.query(models.Simulation).\
            filter(models.Simulation.id.like('%wind_tunnel_discretized_matched%')).all()

        for sim in sims:
            insect_params = {ip.name: ip.value for ip in sim.insect.insect_params}
            w = insect_params['w']
            if '0.3mps' in sim.id:
                self.assertAlmostEqual(w, 0.3, delta=.0001)
            elif '0.4mps' in sim.id:
                self.assertAlmostEqual(w, 0.4, delta=.0001)
            elif '0.6mps' in sim.id:
                self.assertAlmostEqual(w, 0.6, delta=.0001)

    def test_starting_position_idxs_are_correct(self):
        sims = session.query(models.Simulation).\
            filter(models.Simulation.id.like('%wind_tunnel_discretized_matched%')).all()

        for sim in sims:
            for trial in sim.trials:
                first_tp = trial.get_timepoints(session).first()
                trial_start_idx = (first_tp.xidx, first_tp.yidx, first_tp.zidx)
                self.assertEqual(trial_start_idx, trial.geom_config.start_idx)


if __name__ == '__main__':
    unittest.main()