import numpy as np


def plume_and_traj_3d(axs, sim):
    """Plot trajectory from simulation."""

    axs[0].cla()
    axs[1].cla()
    # plot cross-section of plume
    axs[0].matshow(sim.pl.concxy.T, origin='lower')
    axs[1].matshow(sim.pl.concxz.T, origin='lower')

    # overlay trajectory
    x = sim.pos_idx[:, 0]
    y = sim.pos_idx[:, 1]
    z = sim.pos_idx[:, 2]

    axs[0].plot(x[:sim.ts], y[:sim.ts], color='k', lw=2)
    axs[1].plot(x[:sim.ts], z[:sim.ts], color='k', lw=2)

    # overlay hits
    if np.any(sim.hits):
        xhit = x[sim.hits > 0]
        yhit = y[sim.hits > 0]
        zhit = z[sim.hits > 0]

        axs[0].scatter(xhit[:sim.ts], yhit[:sim.ts], s=50, c='r')
        axs[1].scatter(xhit[:sim.ts], zhit[:sim.ts], s=50, c='r')


def src_prob_and_traj_3d(axs, sim):
    """Plot trajectory from simulation."""

    axs[0].cla()
    axs[1].cla()
    # plot cross-section of log probability
    axs[0].matshow(sim.ins.logprobxy.T, origin='lower')
    axs[1].matshow(sim.ins.logprobxz.T, origin='lower')

    # overlay trajectory
    x = sim.pos_idx[:, 0]
    y = sim.pos_idx[:, 1]
    z = sim.pos_idx[:, 2]

    axs[0].plot(x[:sim.ts], y[:sim.ts], color='k', lw=2)
    axs[1].plot(x[:sim.ts], z[:sim.ts], color='k', lw=2)

    # overlay hits
    if np.any(sim.hits):
        xhit = x[sim.hits > 0]
        yhit = y[sim.hits > 0]
        zhit = z[sim.hits > 0]

        axs[0].scatter(xhit[:sim.ts], yhit[:sim.ts], s=50, c='r')
        axs[1].scatter(xhit[:sim.ts], zhit[:sim.ts], s=50, c='r')