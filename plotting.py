import numpy as np


def plume_and_traj_3d(axs, trial):
    """Plot trajectory from simulation overlaid on plume."""

    axs[0].cla()
    axs[1].cla()
    # plot cross-section of plume
    axs[0].matshow(trial.pl.concxy.T, origin='lower')
    axs[1].matshow(trial.pl.concxz.T, origin='lower')

    # overlay trajectory
    x = trial.pos_idx[:, 0]
    y = trial.pos_idx[:, 1]
    z = trial.pos_idx[:, 2]

    axs[0].plot(x[:trial.ts], y[:trial.ts], color='k', lw=2)
    axs[1].plot(x[:trial.ts], z[:trial.ts], color='k', lw=2)

    # overlay hits
    if np.any(trial.hits):
        xhit = x[trial.hits > 0]
        yhit = y[trial.hits > 0]
        zhit = z[trial.hits > 0]

        axs[0].scatter(xhit[:trial.ts], yhit[:trial.ts], s=50, c='r')
        axs[1].scatter(xhit[:trial.ts], zhit[:trial.ts], s=50, c='r')


def src_prob_and_traj_3d(axs, trial):
    """Plot trajectory from simulation overlaid on insect's estimate of source
    location probability distribution."""

    axs[0].cla()
    axs[1].cla()
    # plot cross-section of log probability
    axs[0].matshow(trial.ins.logprobxy.T, origin='lower')
    axs[1].matshow(trial.ins.logprobxz.T, origin='lower')

    # overlay trajectory
    x = trial.pos_idx[:, 0]
    y = trial.pos_idx[:, 1]
    z = trial.pos_idx[:, 2]

    axs[0].plot(x[:trial.ts], y[:trial.ts], color='k', lw=2)
    axs[1].plot(x[:trial.ts], z[:trial.ts], color='k', lw=2)

    # overlay hits
    if np.any(trial.hits):
        xhit = x[trial.hits > 0]
        yhit = y[trial.hits > 0]
        zhit = z[trial.hits > 0]

        axs[0].scatter(xhit[:trial.ts], yhit[:trial.ts], s=50, c='r')
        axs[1].scatter(xhit[:trial.ts], zhit[:trial.ts], s=50, c='r')

def plume_traj_and_entropy_3d(axs, trial):
    """Plot trajectory from simulation overlaid on plume, along with entropy
    of source distribution as a function of time since start of search."""

    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    # plot cross-section of plume
    axs[0].matshow(trial.pl.concxy.T, origin='lower')
    axs[1].matshow(trial.pl.concxz.T, origin='lower')

    # overlay trajectory
    x = trial.pos_idx[:, 0]
    y = trial.pos_idx[:, 1]
    z = trial.pos_idx[:, 2]

    axs[0].plot(x[:trial.ts], y[:trial.ts], color='k', lw=2)
    axs[1].plot(x[:trial.ts], z[:trial.ts], color='k', lw=2)

    # plot entropy
    ts = np.arange(len(x))
    axs[2].plot(ts[:trial.ts], trial.entropies[:trial.ts], lw=2)

    # overlay hits
    if np.any(trial.hits):
        xhit = x[trial.hits > 0]
        yhit = y[trial.hits > 0]
        zhit = z[trial.hits > 0]

        ts_hit = ts[sim.hits > 0]
        entropies_hit = trial.entropies[trial.hits > 0]

        axs[0].scatter(xhit[:trial.ts], yhit[:trial.ts], s=50, c='r')
        axs[1].scatter(xhit[:trial.ts], zhit[:trial.ts], s=50, c='r')
        axs[2].scatter(ts_hit[:trial.ts], entropies_hit[:trial.ts], s=50, c='r')

    # label axes
    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('source position entropy')