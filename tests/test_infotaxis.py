import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from plot import set_font_size


def test_get_p_src_found_returns_correct_probabilities_for_examples():
    """
    Test function that determines the probability that the source is within
    a certain radius of a position.
    """
    from infotaxis import get_p_src_found

    xs = np.arange(6)
    ys = np.arange(4)

    # test with uniform distribution
    p_src = np.ones((6, 4)) / 24
    log_p_src = np.log(p_src)

    # first test position and radius
    pos_0 = (4, 2)
    radius_0 = 1.05

    p_src_found_correct = 5/24
    p_src_found = get_p_src_found(pos_0, xs, ys, log_p_src, radius_0)

    assert round(p_src_found - p_src_found_correct, 6) == 0

    # second test position
    pos_1 = (1, 0)

    p_src_found_correct = 4/24
    p_src_found = get_p_src_found(pos_1, xs, ys, log_p_src, radius_0)

    assert round(p_src_found - p_src_found_correct, 6) == 0

    # second test radius
    radius_1 = 1.43

    p_src_found_correct = 6/24
    p_src_found = get_p_src_found(pos_1, xs, ys, log_p_src, radius_1)

    assert round(p_src_found - p_src_found_correct, 6) == 0


def test_get_p_sample_works_qualitatively_for_examples():
    """
    Make sure that when src is well localized we get a higher sampling rate just
    downwind of src than farther downwind of src.
    """
    from infotaxis import build_log_src_prior, get_p_sample

    xs = np.linspace(0, 2, 50)
    ys = np.linspace(0, 1, 25)

    dt = 0.1
    w = 0.5
    d = 0.02
    r = 100
    a = 0.003
    tau = 1000

    # make source be localized to a small upwind region
    xs_, ys_ = np.meshgrid(xs, ys, indexing='ij')
    log_p_src = build_log_src_prior('uniform', xs, ys)
    mask = (xs_ >= 0.1) * (xs_ < 0.2) * (ys_ >= 0.6) * (ys_ < 0.7)
    log_p_src[~mask] = -np.inf

    # consider a position near to the source and one far away
    pos_near = (0.3, 0.65)
    pos_far = (1, 0.65)

    p_miss_near = get_p_sample(
        pos=pos_near, h=0, xs=xs, ys=ys, dt=dt,
        w=w, d=d, r=r, a=a, tau=tau, log_p_src=log_p_src)
    p_hit_near = get_p_sample(
        pos=pos_near, h=1, xs=xs, ys=ys, dt=dt,
        w=w, d=d, r=r, a=a, tau=tau, log_p_src=log_p_src)
    p_miss_far = get_p_sample(
        pos=pos_far, h=0, xs=xs, ys=ys, dt=dt,
        w=w, d=d, r=r, a=a, tau=tau, log_p_src=log_p_src)
    p_hit_far = get_p_sample(
        pos=pos_far, h=1, xs=xs, ys=ys, dt=dt,
        w=w, d=d, r=r, a=a, tau=tau, log_p_src=log_p_src)

    # make sure hit and miss probabilities sum to 1
    assert round(p_miss_near + p_hit_near, 5) == 1
    assert round(p_miss_far + p_hit_far, 5) == 1

    # make sure miss detection is lower near the src
    assert p_miss_near < p_miss_far
    # make sure hit detection is higher near the src
    assert p_hit_near > p_hit_far


def show_hit_rate_map_for_example_parameters(
        pos=(1.5, 0.7), w=0.5, d_low=0.02, d_high=0.2, r=100, a=0.003, tau=1000):
    """
    Plot the expected hit rates over a grid of source locations for a given sample
    position. Show results for a low and a high diffusivity coefficient.

    Hit rates should be highest at and upwind of the sample position.
    """
    from infotaxis import get_hit_rate

    xs_src = np.linspace(0, 2, 50)
    ys_src = np.linspace(0, 1, 25)

    dx = np.diff(xs_src).mean()
    dy = np.diff(ys_src).mean()

    extent_x = [xs_src[0] - dx/2, xs_src[-1] + dx/2]
    extent_y = [ys_src[0] - dy/2, ys_src[-1] + dy/2]
    extent = extent_x + extent_y

    axs = plt.subplots(3, 2, figsize=(14, 12), tight_layout=True)[1]

    for d, label, ax_col in zip([d_low, d_high], ['low', 'high'], axs.T):

        hit_rate = get_hit_rate(
            xs_src=xs_src, ys_src=ys_src, pos=pos, w=w,
            d=d, r=r, a=a, tau=tau)

        # hit rate map
        ax_col[0].imshow(
            hit_rate.T, origin='lower', extent=extent, zorder=0, cmap='hot')
        # sample position
        ax_col[0].scatter(
            pos[0], pos[1], color='g', marker='D', s=100, lw=0, zorder=1)
        # wind direction
        ax_col[0].arrow(
            .5, .2, 1, 0, head_width=.05, head_length=.05, lw=4, fc='w', ec='w')

        ax_col[0].set_xlabel('x_src (m)')
        ax_col[0].set_ylabel('y_src (m)')

        ax_col[0].set_title('{} D (D = {})'.format(label, d))

        # 1-D x-slice through hit rate map aligned with sample position
        y_idx = np.argmin(np.abs(pos[1] - ys_src))
        handles = []

        for ctr, color in enumerate((.2, .4, .6, .8)):
            y_idx_ = y_idx + ctr
            hit_rate_ = hit_rate[:, y_idx_]
            color_ = (1 - color, 0, 0)

            label = 'y_src = {0:.3f} m'.format(ys_src[y_idx_])

            handles.append(ax_col[1].plot(
                xs_src, hit_rate_, lw=3, color=color_, label=label)[0])

        ax_col[1].set_xlabel('x_src (m)')
        ax_col[1].set_ylabel('hit rate (Hz)')
        ax_col[1].legend(handles=handles, loc='upper left')

        # 1-D y-slice through hit rate map aligned with sample position
        x_idx = np.argmin(np.abs(pos[0] - xs_src))
        handles = []

        for ctr, color in enumerate((.2, .4, .6, .8)):
            x_idx_ = x_idx + ctr
            hit_rate_ = hit_rate[x_idx_, :]
            color_ = (1 - color, 0, 0)

            label = 'x_src = {0:.3f} m'.format(xs_src[x_idx_])

            handles.append(ax_col[2].plot(
                ys_src, hit_rate_, lw=3, color=color_, label=label)[0])

        ax_col[2].set_xlabel('y_src (m)')
        ax_col[2].set_ylabel('hit rate (Hz)')
        ax_col[2].legend(handles=handles, loc='upper left')

    for ax in axs.flatten():
        set_font_size(ax, 14)


def show_update_log_p_src_gives_correct_qualitative_behavior_for_examples(
        pos=(1.5, 0.7), dt=0.1, w=0.5,
        d=0.05, r=100, a=0.003, tau=1000, src_radius=0.02):
    """
    Plot the resulting source posteriors (where prior is uniform) following
    miss and hit at given sampling position.
    """
    from infotaxis import build_log_src_prior, update_log_p_src

    xs = np.linspace(0, 2, 101)
    ys = np.linspace(0, 1, 51)

    log_src_prior = build_log_src_prior('uniform', xs, ys)
    src_prior = np.exp(log_src_prior)

    # compute posterior after miss
    log_p_src_miss = update_log_p_src(
        pos=pos, xs=xs, ys=ys, dt=dt, h=0, w=w,
        d=d, r=r, a=a, tau=tau, src_radius=src_radius, log_p_src=log_src_prior)
    p_src_miss = np.exp(log_p_src_miss)
    p_src_miss /= p_src_miss.sum()

    # compute posterior after hit
    log_p_src_hit = update_log_p_src(
        pos=pos, xs=xs, ys=ys, dt=dt, h=1, w=w,
        d=d, r=r, a=a, tau=tau, src_radius=src_radius, log_p_src=log_src_prior)
    p_src_hit = np.exp(log_p_src_hit)
    p_src_hit /= p_src_hit.sum()

    # plot prior, posterior after miss, and posterior after hit
    dx = np.diff(xs).mean()
    dy = np.diff(ys).mean()

    extent_x = [xs[0] - dx/2, xs[-1] + dx/2]
    extent_y = [ys[0] - dy/2, ys[-1] + dy/2]
    extent = extent_x + extent_y

    axs = plt.subplots(3, 1, figsize=(7, 10), tight_layout=True)[1]

    axs[0].imshow(src_prior.T, origin='lower', extent=extent, cmap='hot')
    axs[1].imshow(p_src_miss.T, origin='lower', extent=extent, cmap='hot')
    axs[2].imshow(p_src_hit.T, origin='lower', extent=extent, cmap='hot')

    for ax in axs:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    axs[0].set_title('prior')
    axs[1].set_title('posterior after miss')
    axs[2].set_title('posterior after hit')

    for ax in axs.flatten():
        set_font_size(ax, 14)


def show_infotaxis_demo(
        seed=0, grid=(101, 51), src_pos=(.1, .5), start_pos=(1.9, .9),
        dt=.1, speed=.2, max_dur=40, th=.5, src_radius=.02,
        w=.5, d=.05, r=5, a=.003, tau=100):
    """
    Run a quick infotaxis demo and plot the resulting trajectory.
    """
    from infotaxis import simulate
    from plume_processing import IdealInfotaxisPlume
    np.random.seed(seed)

    plume = IdealInfotaxisPlume(
        src_pos=src_pos, w=w, d=d, r=r, a=a, tau=tau, dt=dt)

    # run infotaxis simulation
    traj, hs, src_found, log_p_srcs = simulate(
        plume=plume, grid=grid, start_pos=start_pos, speed=speed, dt=dt,
        max_dur=max_dur, th=th, src_radius=src_radius, w=w, d=d, r=r, a=a, tau=tau,
        return_log_p_srcs=True)

    if src_found:
        print('Source found after {} time steps ({} s)'.format(
            len(traj), len(traj) * dt))
    else:
        print('Source not found after {} time steps ({} s)'.format(
            len(traj), len(traj) * dt))

    # plot trajectory
    gs = gridspec.GridSpec(5, 2)
    fig, axs = plt.figure(figsize=(15, 16), tight_layout=True), []

    # plot full trajectory overlaid on plume profile
    conc, extent = plume.get_profile(grid)
    ax_main = fig.add_subplot(gs[:2, :])
    ax_main.imshow(conc.T, origin='lower', extent=extent, cmap='hot', zorder=0)

    # plot trajectory and hits
    ax_main.plot(traj[:, 0], traj[:, 1], lw=2, color='w', zorder=1)
    ax_main.scatter(
        traj[hs > 0, 0], traj[hs > 0, 1], marker='D', s=50, c='c', zorder=2)
    # mark starting position
    ax_main.scatter(*start_pos, s=30, c='b', zorder=2)
    # mark source location
    ax_main.scatter(*plume.src_pos, marker='*', s=100, c='k', zorder=2)

    # set figure axis limits
    ax_main.set_xlim(extent[:2])
    ax_main.set_ylim(extent[2:])

    # make figure labels
    ax_main.set_xlabel('x (m)')
    ax_main.set_ylabel('y (m)')
    ax_main.set_title('full trajectory with plume profile')

    # plot trajectory and src posterior for 6 time points
    axs = np.empty((3, 2), dtype=object)
    for row in range(3):
        for col in range(2):
            axs[row, col] = fig.add_subplot(gs[2 + row, col])

    # figure out indices of time points to show
    intvl = int(len(traj) / 6)

    for ctr, ax in enumerate(axs.flatten()):

        # get traj/hits/src posterior until current time step index
        t_idx = ctr * intvl
        traj_ = traj[:t_idx]
        hs_ = hs[:t_idx]

        # plot source posterior
        p_src = np.exp(log_p_srcs[t_idx])
        p_src /= p_src.sum()
        ax.imshow(
            p_src.T, origin='lower', extent=plume.x_bounds+plume.y_bounds,
            cmap='hot', zorder=0)

        # plot trajectory
        ax.plot(traj_[:, 0], traj_[:, 1], lw=2, color='w', zorder=1)
        # plot hits
        ax.scatter(
            traj_[hs_ > 0, 0], traj_[hs_ > 0, 1],
            marker='D', lw=0, s=50, c='c', zorder=2)

        # plot starting position
        ax.scatter(*start_pos, s=30, c='b', zorder=2)
        # plot src location
        ax.scatter(*plume.src_pos, marker='*', s=100, c='k', zorder=2)

        # set axis limits
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])

        # make axis labels
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('t = {0:.3f} s'.format(t_idx * dt))

    for ax in [ax_main] + list(axs.flatten()):
        set_font_size(ax, 14)
