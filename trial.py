import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as smooth
from math_tools.fun import cdiff
from math_tools.physics import heading


class Trial(object):

    def __init__(self, pl, ins, nsteps=1000):

        self.pl = pl
        self.ins = ins

        self.ts = -1  # time step

        self.at_src = False

        # allocate arrays for storing data
        self.pos = np.zeros((nsteps, 3), dtype=float)
        self.pos_idx = np.zeros((nsteps, 3), dtype=int)
        self.odor = np.zeros((nsteps,), dtype=float)
        self.detected_odor = np.zeros((nsteps,), dtype=float)
        self.entropies = np.zeros((nsteps,), dtype=float)
        self.dist_to_src = np.zeros((nsteps,), dtype=float)

        self._hxyz = None

        self.start_tp_id, self.end_tp_id = None, None

        self.orm = None

        # take first step
        self.step(first_step=True)

    def step(self, first_step=False):

        if not first_step:
            self.ins.calc_util()
            self.ins.move()

        # get concentration and odor
        odor = self.pl.conc[tuple(self.ins.pos_idx)]
        detected_odor = self.pl.sample(self.ins.pos_idx)

        # let insect sample odor
        self.ins.odor = detected_odor
        # self.ins.sample(detected_odor)

        # update source probability
        self.ins.update_src_prob()

        # update plume
        self.pl.update()

        self.ts += 1

        # calculate distance to source
        self.dist_to_src[self.ts] = np.sum(np.abs(np.subtract(self.ins.pos_idx,
                                                              self.pl.src_pos_idx)))

        # check if insect has found source
        if self.dist_to_src[self.ts] == 0:
            self.at_src = True
            odor = -1
            self.ins.odor = -1
            self.ins.S = 0

        # store all data
        self.pos[self.ts, :] = self.ins.pos
        self.pos_idx[self.ts, :] = self.ins.pos_idx
        self.odor[self.ts] = odor
        self.detected_odor[self.ts] = self.ins.odor
        self.entropies[self.ts] = self.ins.S

    @property
    def hxyz(self):
        if not self._hxyz:
            # get velocity then heading
            v = cdiff(self.pos[:self.ts + 1, :], axis=0)
            self._hxyz = heading(v, up_dir=np.array([-1., 0, 0]), in_deg=True)
        return self._hxyz

    def add_timepoints(self, models, session, heading_smoothing=1):

        # get smoothed headings
        hxyz = smooth(self.hxyz, heading_smoothing)

        # add timepoints
        for tp_ctr in xrange(self.ts + 1):

            tp = models.Timepoint()

            tp.hxyz = hxyz[tp_ctr]
            tp.xidx, tp.yidx, tp.zidx = self.pos_idx[tp_ctr]
            tp.odor = self.odor[tp_ctr]
            tp.detected_odor = self.detected_odor[tp_ctr]
            tp.src_entropy = self.entropies[tp_ctr]
            session.add(tp)

            # get timepoint start and end ids if first iteration
            if tp_ctr == 0:
                session.flush()
                self.start_tp_id = tp.id
                self.end_tp_id = self.start_tp_id + self.ts

    def set_orm(self, models):
        self.info_orm = models.TrialInfo()
        self.info_orm.duration = self.ts + 1
        self.info_orm.found_src = self.at_src

        self.orm = models.Trial()
        self.orm.start_timepoint_id = self.start_tp_id
        self.orm.end_timepoint_id = self.end_tp_id
        self.orm.trial_info = self.info_orm