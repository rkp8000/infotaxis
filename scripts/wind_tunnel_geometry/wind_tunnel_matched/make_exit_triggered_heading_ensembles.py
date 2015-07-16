from __future__ import print_function, division

SCRIPT_ID = 'make_exit_triggered_heading_ensembles'
SCRIPT_NOTES = 'Run for all odor states and experiments with r=1000 and d=0.06, keeping only ' \
               'early (1st or 2nd) cw ' \
               'crossings occurring in middle of wind tunnel.'
import numpy as np
from scipy import stats

from db_api import models, add_script_execution
from db_api.connect import session
from sqlalchemy import and_

from config import *
from config.make_exit_triggered_heading_ensembles import *


def main():
    add_script_execution(SCRIPT_ID, session, multi_use=True, notes=SCRIPT_NOTES)

    for sg_id_template in SEGMENT_GROUP_IDS:
        for expt in EXPERIMENTS:
            for odor_state in ODOR_STATES:
                sg_id = sg_id_template.format(expt, odor_state, 3)
                segment_group = session.query(models.SegmentGroup).get(sg_id)

                print(sg_id)

                # build filter conditions
                filter_conditions = []
                for k, v in CONDITIONS.items():
                    if v is None:
                        continue

                    if k == 'encounter_number_max':
                        filter_conditions += [models.Segment.encounter_number <= v]
                    elif k == 'encounter_number_min':
                        filter_conditions += [models.Segment.encounter_number >= v]
                    elif k == 'heading_max':
                        filter_conditions += [models.Segment.heading_exit < v]
                    elif k == 'heading_min':
                        filter_conditions += [models.Segment.heading_exit >= v]
                    elif k == 'x_idx_max':
                        filter_conditions += [models.Segment.x_idx_exit <= v]
                    elif k == 'x_idx_min':
                        filter_conditions += [models.Segment.x_idx_exit >= v]

                # get all segments
                segs = session.query(models.Segment).\
                    filter(models.Segment.segment_group_id == sg_id).\
                    filter(and_(*filter_conditions))

                # skip if no segments
                if len(segs.all()):

                    # create list of arrays containing the heading data
                    headings = []
                    for seg in segs:
                        tps = seg.fetch_timepoints(session, 'exit', 'end')
                        h = [tp.hxyz for tp in tps[:TIMESTEP_MAX]]

                        diff = TIMESTEP_MAX - len(h)
                        if diff > 0:
                            # pad with nans if necessary
                            h += [np.nan for _ in xrange(diff)]

                        headings += [h]

                    headings = np.array(headings, dtype=float)

                    # get mean, std, sem, and number of segments
                    mean = stats.nanmean(headings, axis=0)
                    std = stats.nanstd(headings, axis=0)

                    sem = []
                    for tp in range(TIMESTEP_MAX):
                        headings_not_nan = headings[~np.isnan(headings[:, tp]), tp]

                        sem += [stats.sem(headings_not_nan)]

                    n_segments = (~np.isnan(headings)).sum(axis=0)

                # create new ensemble data model
                ensemble = models.SegmentGroupAnalysisTriggeredEnsemble()
                ensemble.variable = 'heading'
                ensemble.trigger_start = 'exit'
                ensemble.trigger_end = 'end'
                ensemble.conditions = CONDITIONS

                ensemble.segment_group = segment_group

                if len(segs.all()):
                    ensemble.store_data(session, mean=mean, std=std, sem=sem, n_segments=n_segments)

                session.add(ensemble)

    session.commit()

if __name__ == '__main__':
    main()