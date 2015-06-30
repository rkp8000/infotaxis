from __future__ import division, print_function

SCRIPT_ID = 'make_segment_groups_and_segments'
SCRIPT_NOTES = 'run for all experiments and odor states, for infotaxis and empirical using a heading smoothing of 3 and threshold of 10'

import numpy as np
from scipy.ndimage import gaussian_filter1d as smooth
from math_tools import signal

from db_api import models, add_script_execution
from db_api.connect import session

from config import *
from config.make_segment_groups_and_segments import *


def get_segments(odor, th, idxs, x_idx, heading):
    """
    Break up a time-series into segments and return the segments.
    :param odor: odor time-series
    :param th: threshold for determining segment onset and offset
    :param idxs: indices of time points
    :param x_idx: x-position indices
    :param heading: headings
    :return: list of segments
    """

    seg_data = signal.segment_by_threshold(odor, th=th, seg_start='last', seg_end='next', idxs=idxs)

    starts, onsets, offsets, ends = seg_data.T

    start_trial = idxs[0]

    # add all segments
    segments = []
    for s_ctr in range(len(starts)):
        start = starts[s_ctr]
        onset = onsets[s_ctr]
        offset = offsets[s_ctr]
        end = ends[s_ctr]

        segment = models.Segment()

        # fill in the info for the segment
        segment.timepoint_id_start = start
        segment.timepoint_id_enter = onset
        segment.timepoint_id_exit = offset
        segment.timepoint_id_end = end

        # get basic features
        segment.encounter_number = s_ctr + 1

        segment.heading_enter = heading[onset - start_trial]
        segment.heading_exit = heading[offset - start_trial]

        segment.x_idx_enter = x_idx[onset - start_trial]
        segment.x_idx_exit = x_idx[offset - start_trial]

        segments += [segment]

    return segments


def main():
    add_script_execution(SCRIPT_ID, session, multi_use=True, notes=SCRIPT_NOTES)

    for sim_id_template, sg_id_template in zip(SIM_ID_TEMPLATES, SG_ID_TEMPLATES):
        for expt in EXPERIMENTS:
            for odor_state in ODOR_STATES:

                sim_id = sim_id_template.format(expt, odor_state)
                sim = session.query(models.Simulation).get(sim_id)

                print(sim_id)

                # make new segment group
                sg_id = sg_id_template.format(expt, odor_state, HEADING_SMOOTHING)
                segment_group = models.SegmentGroup(id=sg_id)
                segment_group.heading_smoothing = HEADING_SMOOTHING
                segment_group.threshold = THRESHOLD
                segment_group.simulation = sim

                session.add(segment_group)

                # make segments
                for trial in sim.trials:
                    # get all timepoints
                    tps = trial.get_timepoints(session)

                    odor = np.array([tp.odor for tp in tps])
                    x_idx = np.array([tp.xidx for tp in tps])
                    heading = smooth(np.array([tp.hxyz for tp in tps]), HEADING_SMOOTHING)
                    idxs = np.array([tp.id for tp in tps])

                    # get segment start and end times, etc.
                    segments = get_segments(odor, th=10, idxs=idxs, x_idx=x_idx, heading=heading)

                    for segment in segments:
                        segment.trial = trial
                        segment.segment_group = segment_group
                        session.add(segment)


    session.commit()


if __name__ == '__main__':
    main()