import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal


class IsingDemon:

    def __init__(self, N):
        self.N = N
        self.demon_energy = 0
        self.system_energy = 0
        self.magnetization = 0

        self.mcs = 0
        self.lattice = np.zeros([self.N, self.N])

    def initialize_lattice(self):
        """ Randomly initialize a lattice to values of -1 and 1 """
        random_array = np.random.rand(self.N, self.N)
        self.lattice = np.where(random_array >= 0.5, -1, 1)

        self.system_energy = np.sum(self.lattice)
        self.magnetization = self.system_energy

    def reset(self):
        """ Resets the simulation """
        self.system_energy = 0
        self.demon_energy = 0
        self.magnetization = 0
        self.mcs = 0

        self.lattice = np.zeros([self.N, self.N])

    def perterb(self):
        """ Randomly perturbs a sight, and computes its change in energy """
        x, y = self._pick_random_site()

        # Flip the spin and compute the change in energy, delta E
        new_spin = -1 * self.lattice[y, x]
        nn = self._sum_nearest_neighbors((x, y))
        dE = 2 * new_spin + nn

        # System gives the energy to the demon, accepts change
        if dE <= 0:
            self.demon_energy += dE
            self.system_energy -= dE
            self.magnetization += 2 * new_spin
            self.lattice[y, x] = new_spin

        # System takes the energy from the demon, accepts change
        elif 0 < dE <= self.demon_energy:
            self.demon_energy -= dE
            self.system_energy += dE
            self.magnetization += 2 * new_spin
            self.lattice[y, x] = new_spin

        self.mcs += 1

    def _sum_nearest_neighbors(self, point):
        """ Computes the nearest neighbors of a point """
        x, y = point
        nn_sum = 0

        if x + 1 in range(0, self.N):
            nn_sum += self.lattice[y, x + 1]
        else:
            nn_sum += self.lattice[y, 0]

        if x - 1 in range(0, self.N):
            nn_sum += self.lattice[y, x - 1]
        else:
            nn_sum += self.lattice[y, self.N - 1]

        if y + 1 in range(0, self.N):
            nn_sum += self.lattice[y + 1, x]
        else:
            nn_sum += self.lattice[0, x]

        if y - 1 in range(0, self.N):
            nn_sum += self.lattice[y - 1, x]
        else:
            nn_sum += self.lattice[self.N - 1, x]

        return nn_sum

    def _pick_random_site(self):
        """ Returns a random point on the lattice """
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        return i, j


ising_demon_model = IsingDemon(100)
ising_demon_model.initialize_lattice()

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(ising_demon_model.lattice, cmap='Greys')
ax[0, 1].plot(ising_demon_model.mcs, ising_demon_model.magnetization)
ax[1, 0].plot(ising_demon_model.mcs, ising_demon_model.demon_energy)
ax[1, 1].plot(ising_demon_model.mcs, ising_demon_model.system_energy)

def animate(i):
    ising_demon_model.perterb()
    ax[0, 0].imshow(ising_demon_model.lattice)
    ax[0, 1].plot(ising_demon_model.mcs, ising_demon_model.magnetization, 'b.')
    ax[1, 0].plot(ising_demon_model.mcs, ising_demon_model.demon_energy, 'b.')
    ax[1, 1].plot(ising_demon_model.mcs, ising_demon_model.system_energy, 'b.')


anim = FuncAnimation(fig, animate, interval=1, frames=1000, repeat=False, blit=False)
plt.show()
pass

# https://stackoverflow.com/questions/28074461/animating-growing-line-plot-in-python-matplotlib
#
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
#
# fig, ax = plt.subplots()
# line, = ax.plot(x, y, color='k')
#
# def update(num, x, y, line):
#     line.set_data(x[:num], y[:num])
#     line.axes.axis([0, 10, 0, 1])
#     return line,
#
# ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, y, line],
#                               interval=25, blit=True)
# ani.save('test.gif')
# plt.show()
