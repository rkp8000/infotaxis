from __future__ import division
import unittest

from db_api.connect import session
from db_api import models


class TruismTestCase(unittest.TestCase):

    def test_truisms(self):

        self.assertTrue(True)


class ModelTestCase(unittest.TestCase):

    def test_geom_config_real_trajectory_extension(self):

        gcrte = models.GeomConfigRealTrajectoryExtension()
        geom_config0 = session.query(models.GeomConfig).first()

        gcrte.geom_config = geom_config0
        gcrte.trajectory_id = 'test_trajectory_id'

        session.add(gcrte)

        self.assertTrue(gcrte in geom_config0.geom_config_real_trajectory_extensions)
        self.assertEqual(geom_config0.geom_config_real_trajectory_extensions[-1].trajectory_id,
                         'test_trajectory_id')

        session.rollback()


if __name__ == '__main__':
    unittest.main()