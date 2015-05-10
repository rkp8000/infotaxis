import numpy as np


class Trial(object):

    def __init__(self, pl, ins, nsteps=1000):

        self.pl = pl
        self.ins = ins

        self.ts = -1  # time step

        self.at_src = False

        # allocate arrays for storing data
        self.pos = np.zeros((nsteps, 3), dtype=float)
        self.pos_idx = np.zeros((nsteps, 3), dtype=int)
        self.conc = np.zeros((nsteps,), dtype=float)
        self.odor = np.zeros((nsteps,), dtype=float)
        self.hits = np.zeros((nsteps,), dtype=float)
        self.entropies = np.zeros((nsteps,), dtype=float)
        self.dist_to_src = np.zeros((nsteps,), dtype=float)

        self.start_tp_id, self.end_tp_id = None, None

        self.orm = None

        # take first step
        self.step(first_step=True)

    def step(self, first_step=False):

        if not first_step:
            self.ins.calc_util()
            self.ins.move()

        # get concentration and odor
        conc = self.pl.conc[tuple(self.ins.pos_idx)]
        odor = self.pl.sample(self.ins.pos_idx)

        # let insect sample odor
        self.ins.sample(odor)

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

        # store all data
        self.pos[self.ts, :] = self.ins.pos
        self.pos_idx[self.ts, :] = self.ins.pos_idx
        self.conc[self.ts] = conc
        self.odor[self.ts] = self.ins.odor
        self.hits[self.ts] = self.ins.hit
        self.entropies[self.ts] = self.ins.S

    @property
    def hxy(self):
        return 3*np.ones((nsteps,))

    @property
    def hxyz(self):
        return 3*np.ones((nsteps,))

    def add_timepoints(self, models, session):

        # add timepoints
        for tp_ctr in xrange(self.ts + 1):
            tp = models.Timepoint()
            tp.xidx, tp.yidx, tp.zidx = self.pos_idx[tp_ctr]
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
        self.orm.trial_info = [self.info_orm]