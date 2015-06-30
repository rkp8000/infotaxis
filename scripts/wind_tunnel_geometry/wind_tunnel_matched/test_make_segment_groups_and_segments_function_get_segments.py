from __future__ import division, print_function
import unittest
import numpy as np

from make_segment_groups_and_segments import get_segments


class TruismsTestCase(unittest.TestCase):

    def test_truisms(self):
        self.assertTrue(True)


class GetSegmentsTestCase(unittest.TestCase):

    def test_segments_made_correctly(self):

        odor = np.array([0, 0, 0, 20, 20, 0, 0, 20, 20, 20, 0, 0, 0, 20, 20])
        x_idx = -np.arange(15)
        heading = np.arange(15)

        idxs = np.arange(15) + 10
        segments = get_segments(odor, th=10, idxs=idxs, x_idx=x_idx, heading=heading)

        self.assertEqual(len(segments), 3)

        timepoints_correct = [(10, 13, 14, 16),
                              (15, 17, 19, 22),
                              (20, 23, 24, 24)]
        heading_enter_correct = [3, 7, 13]
        heading_exit_correct = [4, 9, 14]
        x_idx_enter_correct = [-3, -7, -13]
        x_idx_exit_correct = [-4, -9, -14]

        for s_ctr, s in enumerate(segments):
            t_segment = (s.timepoint_id_start, s.timepoint_id_enter,
                         s.timepoint_id_exit, s.timepoint_id_end)

            self.assertEqual(t_segment, timepoints_correct[s_ctr])

            self.assertEqual(s.heading_enter, heading_enter_correct[s_ctr])
            self.assertEqual(s.heading_exit, heading_exit_correct[s_ctr])

            self.assertEqual(s.x_idx_enter, x_idx_enter_correct[s_ctr])
            self.assertEqual(s.x_idx_exit, x_idx_exit_correct[s_ctr])

            self.assertEqual(s.encounter_number, s_ctr + 1)


if __name__ == '__main__':
    unittest.main()