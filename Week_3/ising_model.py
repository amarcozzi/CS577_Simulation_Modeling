import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal


class IsingDemon:

    def __init__(self, N, target):
        self.N = N
        self.target = target

        self.demon_energy = 0
        self.system_energy = 0
        self.magnetization = 0

        self.system_energy_accum = 0
        self.mag_accum = 0
        self.demon_energy_accum = 0

        self.magnetization_data = list()
        self.system_energy_data = list()
        self.demon_energy_data = list()
        self.temp = list()

        self.mcs = 0
        self.lattice = np.zeros([self.N, self.N], dtype=np.int8)

    def initialize_system(self):
        """ Initalize a system to all + spins, """
        self.lattice = np.ones([self.N, self.N], dtype=np.int8)

        E = -self.N ** 2
        self.magnetization = self.N ** 2

        # Try up to 10*N**2 times to flip spins so that the system has the desired energy
        while E < self.target:
            x, y = self._pick_random_site()
            nn = self._sum_nearest_neighbors((x, y))
            dE = 2 * self.lattice[y, x] * nn

            if dE > 0:
                E += dE
                new_spin = -1 * self.lattice[y, x]
                self.lattice[y, x] = new_spin
                self.magnetization += 2 * new_spin

        self.system_energy = E
        # self.magnetization_data.append(self.magnetization)
        # self.system_energy_data.append(self.system_energy)
        # self.demon_energy_data.append(self.demon_energy)
        # self.temp.append(None)

    def reset(self):
        """ Resets the simulation """
        self.demon_energy = 0
        self.system_energy = 0
        self.magnetization = 0

        self.system_energy_accum = 0
        self.mag_accum = 0
        self.demon_energy_accum = 0

        self.magnetization_data = list()
        self.system_energy_data = list()
        self.demon_energy_data = list()
        self.temp = list()

        self.mcs = 0
        self.lattice = np.zeros([self.N, self.N])

    def perterb(self):
        """ Randomly perturbs N^2 sights, and computes the system's change in energy """
        self.mcs += 1
        for i in range(0, self.N ** 2):
            x, y = self._pick_random_site()

            # Flip the spin and compute the change in energy, delta E
            nn = self._sum_nearest_neighbors((x, y))
            dE = 2 * self.lattice[y, x] * nn

            # System gives the energy to the demon, accepts change
            if dE <= self.demon_energy:
                new_spin = -1 * self.lattice[y, x]
                self.lattice[y, x] = new_spin
                self.demon_energy -= dE
                self.system_energy += dE
                self.magnetization += 2 * new_spin

            # System takes the energy from the demon, accepts change
            # elif 0 < dE <= self.demon_energy:
            #     self.demon_energy -= np.abs(dE)
            #     self.system_energy += np.abs(dE)
            #     self.magnetization += 2 * new_spin
            #     self.lattice[y, x] = new_spin

        self.system_energy_accum += self.system_energy
        self.demon_energy_accum += self.demon_energy
        self.mag_accum += self.magnetization

        self.magnetization_data.append(self.mag_accum / self.mcs)
        self.system_energy_data.append(self.system_energy_accum / self.mcs)
        self.demon_energy_data.append(self.demon_energy_accum / self.mcs)
        self.temp.append(self.temperature())

    def temperature(self):
        return 4.0 / np.log(1.0 + (4.0 * self.mcs * self.N * self.N) / self.demon_energy_accum)

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

    def _sum_lattice_nearest_neighbors(self):
        """ Uses Convolve 2D to sum all of the lattice nearest neighbors (with periodic boundary) """
        k = [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]
             ]
        nn = signal.convolve2d(self.lattice, k, mode='same', boundary='wrap')
        return np.sum(nn)

    def _pick_random_site(self):
        """ Returns a random point on the lattice """
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        return i, j


# Create an Ising model with a Demon
ising_model = IsingDemon(50, -1000)
ising_model.initialize_system()

# Initialize the dashboard
fig, ax = plt.subplots(2, 2)
im = ax[0, 0].imshow(ising_model.lattice, cmap='Greys')
temp_line, = ax[0, 1].plot([], [], lw=3)
mag_line, = ax[1, 0].plot([], [], lw=3)
sys_energy_line, = ax[1, 1].plot([], [], lw=3)
ax[0, 1].set_title('Temperature')
ax[1, 0].set_title('Magnetization')
ax[1, 1].set_title('System Energy')


# Define the animation for the ising/demon model dashboard
def animate(i):
    ising_model.perterb()
    im.set_data(ising_model.lattice)
    x_data = np.arange(1, ising_model.mcs + 1)

    temp_line.set_data(x_data, ising_model.temp)
    mag_line.set_data(x_data, ising_model.magnetization_data)
    sys_energy_line.set_data(x_data, ising_model.system_energy_data)

    ax[0, 0].relim()
    ax[0, 0].autoscale_view(True, True, True)
    ax[0, 1].relim()
    ax[0, 1].autoscale_view(True, True, True)
    ax[1, 0].relim()
    ax[1, 0].autoscale_view(True, True, True)
    ax[1, 1].relim()
    ax[1, 1].autoscale_view(True, True, True)


# Animate the ising demon model
anim = FuncAnimation(fig, animate, interval=1, frames=1000, repeat=False, blit=False)
plt.show()

