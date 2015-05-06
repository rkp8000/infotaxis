"""
Generate a 3D infotaxis trajectory as the insect flies through a basic plume.

The settings and parameters of this demo are located in config/basic_plume_3d.py.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from insect import Insect
from plume import BasicPlume
from simulation import Simulation
from plotting import plume_and_traj_3d as plot_sim

from config.basic_plume_3d import *


# create plume
pl = BasicPlume(env=ENV, dt=DT)
pl.set_aux_params(w=W, r=R, d=D, a=A, tau=TAU)
pl.set_src_pos(SRCPOS)

# create insect
ins = Insect(env=ENV, dt=DT)
ins.extract_plume_params(pl=pl)
ins.loglike_function = LOGLIKE
ins.set_pos(STARTPOS)

# create simulation
pl.initialize()
ins.initialize()
nsteps = int(np.floor(RUNTIME/DT))

sim = Simulation(pl=pl, ins=ins, nsteps=nsteps)

# open figure and axes
_, axs = plt.subplots(2, 1, **PLOTKWARGS)
plt.draw()

# run simulation, plotting along the way if necessary
for step in xrange(nsteps - 1):
    sim.step()
    if (step % PLOTEVERY == 0) or (step == nsteps - 1):
        plot_sim(axs, sim)
        plt.draw()
    if PAUSEEVERY:
        if step % PAUSEEVERY == 0:
            raw_input()