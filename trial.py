import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as smooth
from math_tools.fun import cdiff
from math_tools.physics import heading


class Trial(object):

    def __init__(self, pl, ins, nsteps=1000, orm=None):

        self.pl = pl
        self.ins = ins

        self.ts = -1  # time step

        self.at_src = False

        if orm:
            nsteps = orm.trial_info.duration
        # allocate arrays for storing data
        self.pos = np.zeros((nsteps, 3), dtype=float)
        self.pos_idx = np.zeros((nsteps, 3), dtype=int)
        self.odor = np.zeros((nsteps,), dtype=float)
        self.detected_odor = np.zeros((nsteps,), dtype=float)
        self.entropies = np.zeros((nsteps,), dtype=float)
        self.dist_to_src = np.zeros((nsteps,), dtype=float)

        self._hxyz = None

        self.start_tp_id, self.end_tp_id = None, None

        if orm:
            # initialize this trial from a trial stored in the database
            self.orm = orm
        else:
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
        if self.pl.src_pos_idx:
            self.dist_to_src[self.ts] = np.sum(np.abs(np.subtract(self.ins.pos_idx,
                                                                  self.pl.src_pos_idx)))
        else:
            self.dist_to_src[self.ts] = -1

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

    def bind_timepoints(self, models, session):

        if not self.orm:
            raise ValueError('Please set orm before binding timepoints!')

        stp, etp = self.orm.start_timepoint_id, self.orm.end_timepoint_id

        for ts, tp_id in enumerate(range(stp, etp + 1)):
            tp = session.query(models.Timepoint).get(tp_id)
            self.pos_idx[ts] = tp.xidx, tp.yidx, tp.zidx
            self.pos[ts] = self.pl.env.pos_from_idx(self.pos_idx[ts])
            self.odor[ts] = tp.odor
            self.detected_odor[ts] = tp.detected_odor
            self.entropies[ts] = tp.src_entropy

    def generate_orm(self, models):
        self.orm = models.Trial()
        self.orm.start_timepoint_id = self.start_tp_id
        self.orm.end_timepoint_id = self.end_tp_id
        self.orm.trial_info = models.TrialInfo()
        self.orm.trial_info.duration = self.ts + 1
        self.orm.trial_info.found_src = self.at_src


class TrialFromPositionSequence(Trial):
    """For trials that are constructed by discretizing an empirical trajectory"""

    def __init__(self, positions, pl, ins=None):
        """
        :param positions: N x 3 array of positions
        :param pl: plume object used for discretization
        :param ins: insect object used to calculate source position probabilities
        """

        # call parent __init__ method
        super(TrialFromPositionSequence, self).__init__(pl, ins)

        # discretize trajectory positions and get/set average dt
        self.forced_pos_idxs = self.pl.env.discretize_position_sequence(positions)
        self.avg_dt = (0.01 * len(positions)) / len(self.forced_pos_idxs)
        if ins:
            self.ins.dt = self.avg_dt
            self.ins.initialize()

        # step through all positions and fill in timepoints, etc
        self.step(forced_pos_idx=self.forced_pos_idxs[0], first_step=True)

        for forced_pos_idx in self.forced_pos_idxs[1:]:
            self.step(forced_pos_idx=forced_pos_idx)

    def step(self, forced_pos_idx, first_step=False):

        if not first_step:
            self.ins.calc_util()

        self.ins.move(forced_pos_idx)

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
        if self.pl.src_pos_idx:
            self.dist_to_src[self.ts] = np.sum(np.abs(np.subtract(self.ins.pos_idx,
                                                                  self.pl.src_pos_idx)))
        else:
            self.dist_to_src[self.ts] = -1

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