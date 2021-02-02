import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class IsingDemon:

    def __init__(self, N, target):
        self.N = N
        self.target = target

        self.demon_energy = 0
        self.system_energy = 0
        self.magnetization = 0
        self.temp = 0

        self.system_energy_accum = 0
        self.mag_accum = 0
        self.mag_squared_accum = 0
        self.demon_energy_accum = 0

        self.magnetization_data = list()
        self.mag_squared_data = list()
        self.system_energy_data = list()
        self.demon_energy_data = list()
        self.temp_data = list()

        self.mag_data_avg = list()
        self.mag_squared_data_avg = list()
        self.system_energy_data_avg = list()

        self.mcs = 0
        self.lattice = np.zeros([self.N, self.N], dtype=np.int8)

    def initialize_system(self):
        """ Initialize a system to all + spins, """
        self.lattice = np.ones([self.N, self.N], dtype=np.int8)

        E = -self.N ** 2
        self.magnetization = self.N ** 2

        # Try up to 10*N**2 times to flip spins so that the system has the desired energy
        while E < self.target:
            x, y = self._pick_random_site()
            dE = self._compute_change_energy((x, y))

            if dE > 0:
                E += dE
                new_spin = -1 * self.lattice[y, x]
                self.lattice[y, x] = new_spin
                self.magnetization += 2 * new_spin

        self.system_energy = E

    def perterb(self):
        """ Randomly perturbs N^2 sights, and computes the system's change in energy """
        self.mcs += 1
        for i in range(0, self.N ** 2):
            # Pick a random site to perterb
            x, y = self._pick_random_site()

            # Flip the spin and compute the change in energy, delta E
            dE = self._compute_change_energy((x, y))

            # System gives the energy to the demon, accepts change
            if dE <= self.demon_energy:
                new_spin = -1 * self.lattice[y, x]
                self.lattice[y, x] = new_spin
                self.demon_energy -= dE
                self.system_energy += dE
                self.magnetization += 2 * new_spin

            # Update values
            self.demon_energy_accum += self.demon_energy

        self.system_energy_accum += self.system_energy
        self.mag_accum += self.magnetization
        self.mag_squared_accum += self.magnetization ** 2

        # Add values to lists for plotting
        self.magnetization_data.append(self.magnetization)
        self.mag_squared_data.append(self.magnetization ** 2)
        self.system_energy_data.append(self.system_energy)
        self.demon_energy_data.append(self.demon_energy)
        self.temp_data.append(self._compute_temperature())

        self.mag_data_avg.append(self.mag_accum / self.mcs)
        self.mag_squared_data_avg.append(self.mag_squared_accum / self.mcs)
        self.system_energy_data_avg.append(self.system_energy_accum / self.mcs)

    def _compute_temperature(self):
        """ Computes the temperature of the system """
        return 4.0 / np.log(1.0 + 4.0 / (self.demon_energy_accum / (self.mcs * self.N ** 2)))

    def _compute_change_energy(self, point):
        """ Computes the energy at any point on the lattice """
        x, y = point
        nn = self._sum_nearest_neighbors(point)
        return 2 * self.lattice[y, x] * nn

    def _sum_nearest_neighbors(self, point):
        """ Computes the nearest neighbors of a point """
        x, y = point
        right = self.lattice[y, (x + 1) % self.N]
        left = self.lattice[y, (x - 1) % self.N]
        up = self.lattice[(y - 1) % self.N, x]
        down = self.lattice[(y + 1) % self.N, x]

        return right + left + up + down

    def _pick_random_site(self):
        """ Returns a random point on the lattice """
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        return i, j


# Create an Ising model with a Demon
ising_model = IsingDemon(50, -100)
ising_model.initialize_system()

# Initialize the dashboard
fig, ax = plt.subplots(2, 2)
im = ax[0, 0].imshow(ising_model.lattice, cmap='Greys')
temp_line, = ax[0, 1].plot([], [], lw=3)
mag_line, = ax[1, 0].plot([], [], lw=3)
sys_energy_line, = ax[1, 1].plot([], [], lw=3)
ax[0, 0].set_title('System')
ax[0, 1].set_title('Temperature')
ax[1, 0].set_title('Magnetization')
ax[1, 1].set_title('System Energy')


# Define the animation for the ising/demon model dashboard
def animate(i):
    # Run a step in the simulation and display the simulation image
    ising_model.perterb()
    im.set_data(ising_model.lattice)

    # Plot temp_data, magnetism, and system energy over monte carlo steps
    x_data = np.arange(1, ising_model.mcs + 1)
    temp_line.set_data(x_data, ising_model.temp_data)
    mag_line.set_data(x_data, ising_model.mag_data_avg)
    sys_energy_line.set_data(x_data, ising_model.system_energy_data_avg)

    # Reset the scale and limits of the plots
    ax[0, 0].relim()
    ax[0, 0].autoscale_view(True, True, True)
    ax[0, 1].relim()
    ax[0, 1].autoscale_view(True, True, True)
    ax[1, 0].relim()
    ax[1, 0].autoscale_view(True, True, True)
    ax[1, 1].relim()
    ax[1, 1].autoscale_view(True, True, True)


# Animate the ising demon model
anim = FuncAnimation(fig, animate, interval=1, frames=250, repeat=False)
plt.show()