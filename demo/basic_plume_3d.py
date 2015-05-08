"""
Generate a 3D infotaxis trajectory as the insect flies through a basic plume.

The settings and parameters of this demo are located in config/basic_plume_3d.py.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from insect import Insect
from plume import BasicPlume
from trial import Trial
from plotting import plume_traj_and_entropy_3d as plot_trial

from config.basic_plume_3d import *


# create plume
pl = BasicPlume(env=ENV, dt=DT)
pl.set_params(**PLUME_PARAMS)
pl.set_src_pos(SRCPOS, is_idx=True)

# create insect
ins = Insect(env=ENV, dt=DT)
ins.set_params(**PLUME_PARAMS)
ins.loglike_function = LOGLIKE
ins.set_pos(STARTPOS, is_idx=True)

# create simulation
pl.initialize()
ins.initialize()
nsteps = int(np.floor(RUNTIME/DT))

trial = Trial(pl=pl, ins=ins, nsteps=nsteps)

# open figure and axes
_, axs = plt.subplots(3, 1, **PLOTKWARGS)
plt.draw()

# run simulation, plotting along the way if necessary
for step in xrange(nsteps - 1):
    trial.step()
    if (step % PLOTEVERY == 0) or (step == nsteps - 1):
        plot_trial(axs, trial)
        plt.draw()

    if trial.at_src:
        print 'Found source after {} timesteps.'.format(trial.ts)
        break

    if PAUSEEVERY:
        if step % PAUSEEVERY == 0:
            raw_input('Press enter to continue...')

raw_input()