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