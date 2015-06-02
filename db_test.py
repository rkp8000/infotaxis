from __future__ import division
import unittest

from db_api.connect import session
from db_api import models


class TruismTestCase(unittest.TestCase):

    def test_truisms(self):

        self.assertTrue(True)


class ModelTestCase(unittest.TestCase):

    def test_geom_config_extension_real_trajectory(self):

        gcert = models.GeomConfigExtensionRealTrajectory()
        geom_config0 = session.query(models.GeomConfig).first()

        gcert.geom_config = geom_config0
        gcert.real_trajectory_id = 'test_trajectory_id'

        session.add(gcert)

        self.assertEqual(gcert, geom_config0.geom_config_extension_real_trajectory)
        self.assertEqual(geom_config0.geom_config_extension_real_trajectory.real_trajectory_id,
                         'test_trajectory_id')

        # try querying the join table
        q = session.query(models.GeomConfig, models.GeomConfigExtensionRealTrajectory). \
            filter(models.GeomConfig.id == models.GeomConfigExtensionRealTrajectory.geom_config_id). \
            filter(models.GeomConfigExtensionRealTrajectory.real_trajectory_id == 'test_trajectory_id').all()

        gc, gcert = q[0]
        self.assertEqual(gcert.real_trajectory_id, 'test_trajectory_id')

        session.rollback()


if __name__ == '__main__':
    unittest.main()