"""
Generate a 2D infotaxis trajectory as the insect flies through a basic plume.

The settings and parameters of this demo are located in config/basic_plume_2d.py.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from insect import Insect
from logprob_odor import binary_advec_diff_tavg
from plume import BasicPlume, Environment3d
from trial import Trial
from plotting import plume_and_traj_3d as plot_trial


# PLOTTING
PLOTEVERY = 10
PAUSEEVERY = 50
PLOTKWARGS = {'figsize': (10, 10),
              'facecolor': 'w'}


# SIMULATION
RUNTIME = 100  # (s)


# ENVIRONMENT
DT = .06  # (s)
XRBINS = np.linspace(-.3, 1.0, 66)  # (m)
YRBINS = np.linspace(-.15, .15, 16)  # (m)
ZRBINS = np.linspace(-.15, .15, 2)  # (m)
ENV = Environment3d(XRBINS, YRBINS, ZRBINS)


# PLUME
SRCPOS = (0, 7, 0)
PLUME_PARAMS = {
                'w': 0.4,  # wind (m/s)
                'r': 10,  # source emission rate
                'd': 0.12,  # diffusivity (m^2/s)
                'a': .002,  # searcher size (m)
                'tau': 1000,  # particle lifetime (s)
                }


# INSECT
STARTPOS = (64, 7, 0)
LOGLIKE = binary_advec_diff_tavg


def run():
    import pdb; pdb.set_trace()
    # create plume
    pl = BasicPlume(env=ENV, dt=DT)
    pl.set_params(**PLUME_PARAMS)
    pl.set_src_pos(SRCPOS, is_idx=True)

    # create insect
    ins = Insect(env=ENV, dt=DT)
    ins.set_params(**PLUME_PARAMS)
    ins.loglike_function = LOGLIKE
    ins.set_pos(STARTPOS, is_idx=True)

    # create trial
    pl.initialize()
    ins.initialize()
    nsteps = int(np.floor(RUNTIME/DT))

    trial = Trial(pl=pl, ins=ins, nsteps=nsteps)

    # open figure and axes
    _, axs = plt.subplots(3, 1, **PLOTKWARGS)

    # run trial, plotting along the way if necessary
    for step in xrange(nsteps - 1):
        trial.step()

        if trial.at_src:
            print 'Found source!'
            break

    plot_trial(axs, trial)
