from __future__ import print_function, division

TRAJ_LIMIT = 10

import unittest
import generate_trials_one_for_one

from db_api import models
from db_api.connect import session


class MainTestCase(unittest.TestCase):

    def setUp(self):
        try:
            generate_trials_one_for_one.main(TRAJ_LIMIT)
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


if __name__ == '__main__':
    unittest.main()