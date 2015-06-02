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
        gcrte.real_trajectory_id = 'test_trajectory_id'

        session.add(gcrte)

        # make sure geom_config_real_trajectory_extension gets added correctly
        self.assertTrue(gcrte in geom_config0.geom_config_real_trajectory_extensions)
        self.assertEqual(geom_config0.geom_config_real_trajectory_extensions[-1].real_trajectory_id,
                         'test_trajectory_id')

        # try querying the join table
        q = session.query(models.GeomConfig, models.GeomConfigRealTrajectoryExtension). \
            filter(models.GeomConfig.id == models.GeomConfigRealTrajectoryExtension.geom_config_id). \
            filter(models.GeomConfigRealTrajectoryExtension.real_trajectory_id == 'test_trajectory_id').all()

        gc, gcrte = q[0]
        self.assertEqual(gcrte.real_trajectory_id, 'test_trajectory_id')

        session.rollback()


if __name__ == '__main__':
    unittest.main()