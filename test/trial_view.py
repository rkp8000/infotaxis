import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from insect import Insect
from plume import BasicPlume
from trial import Trial
from plotting import plume_and_traj_3d as plot_trial

from db_api import mappings
from db_api.connect import engine, session, TESTCXN

from config.trial_view import *


# get simulation
sim = session.query(mappings.Simulation).get(SIMULATIONID)
print sim.id

pl = BasicPlume(sim.env, sim.dt, orm=sim.pl)
ins = Insect(sim.env, sim.dt, orm=sim.ins)

pl.initialize()
ins.initialize()

_, axs = plt.subplots(2, 1)

for tr_orm in sim.trials:
    [ax.cla() for ax in axs]

    trial = Trial(pl, ins, orm=tr_orm, session=session)
    plot_trial(axs, trial)
    axs[0].set_title('trial {} from {}'.format(trial.orm.id, sim.id))
    plt.draw()

    raw_input()