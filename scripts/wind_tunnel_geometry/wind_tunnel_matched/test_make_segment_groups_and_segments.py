from __future__ import print_function, division

SIMULATION_ID_EMPIRICAL = 'wind_tunnel_discretized_copies_r1000_d0.02'
SIMULATION_ID_INFOTAXIS = 'wind_tunnel_discretized_matched_r1000_d0.02'

import unittest


from db_api import models
from db_api.connect import session

import make_segment_groups_and_segments


class MainTestCase(unittest.TestCase):

    def test_correct_number_of_segment_groups_with_proper_smoothing(self):

        heading_smoothing = make_segment_groups_and_segments.HEADING_SMOOTHING
        for sim_id in (SIMULATION_ID_EMPIRICAL, SIMULATION_ID_INFOTAXIS):
            segment_groups = session.query(models.SegmentGroup).\
                filter(models.SegmentGroup.simulation_id.like(sim_id + '%')).\
                filter(models.SegmentGroup.heading_smoothing == heading_smoothing)

            self.assertEqual(len(segment_groups.all()), 9)

            for segment_group in segment_groups:
                n_segments = len(segment_group.segments)
                print('{} segments in segment group {}'.format(n_segments, segment_group.id))

if __name__ == '__main__':
    #make_segment_groups_and_segments.main()
    unittest.main()