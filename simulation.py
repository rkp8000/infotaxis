import numpy as np


class Simulation(object):

    def __init__(self, pl, ins, nsteps=1000):

        self.pl = pl
        self.ins = ins

        self.ts = -1  # time step

        # allocate arrays for storing data
        self.pos = np.zeros((nsteps, 3), dtype=float)
        self.pos_idx = np.zeros((nsteps, 3), dtype=int)
        self.conc = np.zeros((nsteps,), dtype=float)
        self.odor = np.zeros((nsteps,), dtype=float)
        self.hits = np.zeros((nsteps,), dtype=float)
        self.entropies = np.zeros((nsteps,), dtype=float)

        # take first step
        self.step(first_step=True)

    def step(self, first_step=False):

        if not first_step:
            self.ins.calc_util()
            self.ins.move()
            print self.ins.pos

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

        print self.ts

        # store all data
        self.pos[self.ts, :] = self.ins.pos
        self.pos_idx[self.ts, :] = self.ins.pos_idx
        self.conc[self.ts] = conc
        self.odor[self.ts] = self.ins.odor
        self.hits[self.ts] = self.ins.hit
        self.entropies[self.ts] = self.ins.S