from __future__ import print_function

TRAJ_LIMIT = 5

import unittest
import generate_discretized_wind_tunnel_trajectory_copies

from db_api.connect import session
from db_api import models


class MainTestCase(unittest.TestCase):

    def setUp(self):
        k0 = generate_discretized_wind_tunnel_trajectory_copies.INSECT_PARAMS_DICT.keys()[0]
        r = generate_discretized_wind_tunnel_trajectory_copies.INSECT_PARAMS_DICT[k0]['r']
        d = generate_discretized_wind_tunnel_trajectory_copies.INSECT_PARAMS_DICT[k0]['d']
        self.sim_id_pattern = 'wind_tunnel_discretized_copies_r{}_d{}_%'.format(r, d)

    def test_correct_number_of_simulations_trials_and_timepoints(self):
        sims = session.query(models.Simulation). \
            filter(models.Simulation.id.like(self.sim_id_pattern))

        self.assertEqual(len(sims.all()), 9)

        for sim in sims:

            self.assertEqual(len(list(sim.trials)), TRAJ_LIMIT)

            for trial in sim.trials:
                # check to make sure trial info duration matches geom_config duration
                self.assertEqual(trial.trial_info.duration, trial.geom_config.duration)
                # check to make sure there are actually as many timepoints connected to the trial
                # as there should be
                self.assertEqual(trial.trial_info.duration,
                                 len(trial.get_timepoints(session).all()))


if __name__ == '__main__':
    generate_discretized_wind_tunnel_trajectory_copies.main(TRAJ_LIMIT)
    unittest.main()