"""Generate infotaxis trials in simulated wind tunnel with simulated collimated plume."""

import matplotlib.pyplot as plt
plt.ion()

from insect import Insect
from plume import CollimatedPlume
from trial import Trial
from plotting import plume_traj_and_entropy_3d as plot_trial

from config.collimated_plume_3d import *

# create plume
pl = CollimatedPlume(env=ENV, dt=DT)
pl.set_params(**PLUMEPARAMS)

# create insect
ins = Insect(env=ENV, dt=DT)
ins.set_params(**INSECTPARAMS)
ins.loglike_function = LOGLIKE

# set insect starting position
ins.set_pos(STARTPOSIDX, is_idx=True)

# initialize plume and insect and create trial
pl.initialize()
ins.initialize()

trial = Trial(pl=pl, ins=ins, nsteps=DURATION)

# open figure and axes
_, axs = plt.subplots(3, 1, **PLOTKWARGS)
plt.draw()

# run trial, plotting along the way if necessary
for step in xrange(DURATION - 1):
    trial.step()
    if (step % PLOTEVERY == 0) or (step == DURATION - 1):
        plot_trial(axs, trial)
        plt.draw()

    if trial.at_src:
        print 'Found source after {} timesteps.'.format(trial.ts)
        break

    if PAUSEEVERY:
        if step % PAUSEEVERY == 0:
            raw_input('Press enter to continue...')

raw_input()