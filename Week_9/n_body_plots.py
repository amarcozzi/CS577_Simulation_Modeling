import matplotlib.pyplot as plt
from matplotlib import animation, rc


def show_anim(t_s, y, y0, d, dt, trace_length=20, out_time=.05):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

    c = ['tab:red', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']
    body_list = []
    trace_list = []

    K = int(out_time / dt)
    t_sd = t_s[::K]
    yd = y[::K, :]

    x_min, x_max, y_min, y_max = 1e9, -1e9, 1e9, -1e9
    for i in range(0, y0.size // d, d):
        x_t = yd[:, i]
        y_t = yd[:, i + 1]
        if x_min > x_t.min():
            x_min = x_t.min()
        if x_max < x_t.max():
            x_max = x_t.max()
        if y_min > y_t.min():
            y_min = y_t.min()
        if y_max < y_t.max():
            y_max = y_t.max()

        ph, = ax.plot(x_t, y_t, '-', color=[.7, .7, .7], linewidth=.7)

    plt.xlim([1.2 * x_min, 1.2 * x_max])
    plt.ylim([1.2 * y_min, 1.2 * y_max])

    ax.axis('off')

    for i in range(0, y0.size // d, d):
        ph, = ax.plot(y0[i], y0[i + 1], 'o', color=c[i // d])
        body_list.append(ph)
        ph, = ax.plot([], [], '-', color=c[i // d])
        trace_list.append(ph)

    def animate(i):
        i = i % (t_sd.size - 1)
        for im, j in zip(body_list, range(0, d * len(body_list), d)):
            im.set_xdata(yd[i + 1, j])
            im.set_ydata(yd[i + 1, j + 1])

        if i > trace_length:
            for im, j in zip(trace_list, range(0, d * len(trace_list), d)):
                im.set_xdata(yd[i - trace_length:i + 1, j])
                im.set_ydata(yd[i - trace_length:i + 1, j + 1])
        return im

    anim = animation.FuncAnimation(fig, animate, interval=20, frames=t_sd.size - 1)
    plt.close(fig)
    return anim


def plot_trajectory(y, y0, d):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    x_min, x_max, y_min, y_max = 1e9, -1e9, 1e9, -1e9
    for i in range(0, y0.size // d, d):
        x_t = y[:, i]
        y_t = y[:, i + 1]
        if x_min > x_t.min():
            x_min = x_t.min()
        if x_max < x_t.max(): x_max = x_t.max()
        if y_min > y_t.min(): y_min = y_t.min()
        if y_max < y_t.max(): y_max = y_t.max()

        ph, = ax.plot(x_t, y_t, '-', color=[.7, .7, .7], linewidth=.5)

    plt.xlim([1.2 * x_min, 1.2 * x_max])
    plt.ylim([1.2 * y_min, 1.2 * y_max])

    ax.axis('off')
    plt.show()


def create_dashboard(h, t, k, p):
    """ Creates a dashboard of plots for time steps, potential, kintetic, and total energy """
    plt.style.use('seaborn')
    # Initialize the dashboard
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Create individual graphs
    dt_line, = ax1.plot(h, lw=3, c='k')
    total_line, = ax2.plot(t, lw=3, c='#d62728')
    k_line, = ax3.plot(k, lw=3, c='#1f77b4')
    p_line = ax4.plot(p, lw=3, c='#2ca02c')

    ax1.set_title(r'Variation in $\Delta t$')
    ax1.set_ylabel(r'$\Delta t$')
    ax2.set_title(r'Total Energy over Time')
    ax2.set_ylabel('Total Energy')
    ax3.set_title('Kinetic Energy over Time')
    ax3.set_ylabel('Kinetic Energy')
    ax3.set_xlabel('Time Steps')
    ax4.set_title('Potential Energy over Time')
    ax4.set_ylabel('Potential Energy')
    ax4.set_xlabel('Time Steps')

    plt.show()

    """im = ax[0, 0].imshow(model.lattice, cmap='Greys', vmin=-1, vmax=1)
    energy_line, = ax[0, 1].plot([], [], lw=3)
    mag_line, = ax[1, 0].plot([], [], lw=3)
    heat_line, = ax[1, 1].plot([], [], lw=3)
    susceptibility_line, = ax[2, 0].plot([], [], lw=3)
    acceptance_line, = ax[2, 1].plot([], [], lw=3)"""

