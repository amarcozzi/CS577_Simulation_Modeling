import matplotlib.pyplot as plt
from matplotlib import animation, rc


def show_anim(t_s, y, y0, d, dt, trace_length=20, out_time=.05):
    plt.style.use('dark_background')
    c = ['tab:red', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']
    body_list = []
    trace_list = []

    K = int(out_time / dt)
    t_sd = t_s[::K]
    yd = y[::K, :]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

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

    # Initialize the dashboard
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    dt_graph = ax[0, 0].plot(h, lw=3)


    """im = ax[0, 0].imshow(model.lattice, cmap='Greys', vmin=-1, vmax=1)
    energy_line, = ax[0, 1].plot([], [], lw=3)
    mag_line, = ax[1, 0].plot([], [], lw=3)
    heat_line, = ax[1, 1].plot([], [], lw=3)
    susceptibility_line, = ax[2, 0].plot([], [], lw=3)
    acceptance_line, = ax[2, 1].plot([], [], lw=3)"""
