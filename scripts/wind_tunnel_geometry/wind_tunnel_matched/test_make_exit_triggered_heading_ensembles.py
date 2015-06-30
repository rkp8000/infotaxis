from __future__ import print_function, division

SG_ID_EMPIRICAL = 'empirical_discretized_fruitfly'
SG_ID_INFOTAXIS = 'infotaxis_wind_tunnel_r1000_d0.12_fruitfly'

import unittest
import numpy as np

import make_exit_triggered_heading_ensembles

from db_api import models
from db_api.connect import session


class MainTestCase(unittest.TestCase):

    def setUp(self):

        self.ensembles_empirical = session.query(models.SegmentGroupAnalysisTriggeredEnsemble). \
            filter(models.SegmentGroupAnalysisTriggeredEnsemble.segment_group_id.
            like(SG_ID_EMPIRICAL + '%'))

        self.ensembles_infotaxis = session.query(models.SegmentGroupAnalysisTriggeredEnsemble). \
            filter(models.SegmentGroupAnalysisTriggeredEnsemble.segment_group_id.
            like(SG_ID_INFOTAXIS + '%'))

    def test_correct_number_of_ensembles_added_and_have_correct_structure(self):

        conditions = make_exit_triggered_heading_ensembles.CONDITIONS

        for ensembles in (self.ensembles_empirical, self.ensembles_infotaxis):
            # only get ensembles that have the correct conditions
            ensembles_kept = []
            for ensemble in ensembles:
                if np.all([ensemble.conditions[key] == conditions[key] for key in conditions.keys()]):
                    ensembles_kept += [ensemble]

            self.assertEqual(len(ensembles_kept), 9)

            for ensemble in ensembles_kept:
                ensemble.fetch_data(session)
                if ensemble._data is None:
                    print(ensemble.segment_group.id)
                    continue
                self.assertEqual(len(ensemble.mean), len(ensemble.std))
                self.assertEqual(len(ensemble.mean), len(ensemble.sem))
                self.assertEqual(len(ensemble.sem), len(ensemble.n_segments))

                self.assertEqual(ensemble._data.size, ensemble.mean.size * 4)


if __name__ == '__main__':
    make_exit_triggered_heading_ensembles.main()
    unittest.main()