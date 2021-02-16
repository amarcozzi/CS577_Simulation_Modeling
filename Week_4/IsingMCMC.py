import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal


class IsingMCMC:
    def __init__(self, N, T=2.0):
        self.critical_temperature = 2.0 / np.log(1.0 + np.sqrt(2.0))
        self.temp = T
        self.N = N
        self.lattice = np.zeros([self.N, self.N])

        # Initialize accumulators to 0
        self.energy = 0
        self.energy_accum = 0
        self.energy_squared_accum = 0
        self.mag = 0
        self.mag_accum = 0
        self.mag_squared_accum = 0

        # Initialize lists to store raw data over time (aka monte carlo steps)
        self.energy_data = list()
        self.mag_data = list()
        self.energy_data_avg = list()
        self.mag_data_avg = list()
        self.specific_heat_data = list()
        self.susceptibility_data = list()
        self.acceptance_rate_data = list()

        self.mcs = 0
        self.accepted_moves = 0
        self.w = dict()

    def initialize_system(self):
        """ Initializes a lattice of size N x_points N to all ones (aka spin up) """
        self.lattice = np.ones([self.N, self.N])  # all spins up
        self.mag = self.N * self.N  # sum of spins
        self.energy = -2 * self.N * self.N  # minimum energy
        # self._reset_data()

        # Boltzmann factors for some reason?
        self.w[8] = np.exp(-8.0 / self.temp)
        self.w[4] = np.exp(-4.0 / self.temp)

    def initialize_random_system(self):
        """ Initializes a lattice of size N x_points N to all ones (aka spin up) """
        random_matrix = np.random.random([self.N, self.N])
        self.lattice = np.where(random_matrix > 0.5, -1, 1)
        self.mag = np.sum(self.lattice)
        k = [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]]
        nn = signal.convolve2d(self.lattice, k, mode='same', boundary='wrap')
        self.energy = -1 * np.sum(np.multiply(self.lattice, nn))
        self._reset_data()

        # Boltzmann factors for some reason?
        self.w[8] = np.exp(-8.0 / self.temp)
        self.w[4] = np.exp(-4.0 / self.temp)

    def _reset_data(self):
        """ Not sure why this is necessary, like many things in modern life, but here we are! """
        self.mcs = 0

        # Initialize accumulators to 0
        self.energy = 0
        self.energy_accum = 0
        self.energy_squared_accum = 0
        self.mag = 0
        self.mag_accum = 0
        self.mag_squared_accum = 0

        # Initialize lists to store raw data over time (aka monte carlo steps)
        self.energy_data = list()
        self.mag_data = list()
        self.energy_data_avg = list()
        self.mag_data_avg = list()
        self.specific_heat_data = list()
        self.susceptibility_data = list()
        self.acceptance_rate_data = list()

    def specific_heat(self):
        """ Computes the specific heat of the system """
        energy_squared_avg = self.energy_squared_accum / self.mcs
        energy_avg = self.energy_accum / self.mcs
        heat_capacity = energy_squared_avg - energy_avg * energy_avg
        heat_capacity /= (self.temp * self.temp)
        return heat_capacity / (self.N * self.N)

    def susceptibility(self):
        """ Computes the susceptibility [?] of the system """
        mag_squared_avg = self.mag_squared_accum / self.mcs
        mag_avg = self.mag_accum / self.mcs
        return (mag_squared_avg - np.square(mag_avg)) / (self.temp * self.N * self.N)

    def do_one_MC_step(self):
        """ Applies the demon algorithm to the heat bath with an embedded ising model """
        # Loop over N^2 iterations for each MC step
        for k in range(self.N * self.N):
            x, y = self._pick_random_site()
            # Flip the spin and compute the change in energy, delta E
            dE = self._compute_change_energy((x, y))

            # Accept the flip and update the lattice, energy, and magnetisation
            if dE <= 0 or self.w.get(int(dE), 0) > np.random.random():
                new_spin = -self.lattice[y, x]
                self.lattice[y, x] = new_spin
                self.accepted_moves += 1
                self.energy += dE
                self.mag += 2 * new_spin

        self.mcs += 1

        # Update accumulators
        self.energy_accum += self.energy
        self.energy_squared_accum += self.energy * self.energy
        self.mag_accum += self.mag
        self.mag_squared_accum += self.mag * self.mag

        # Add values to data storage lists
        self.energy_data.append(self.energy / self.N**2)
        self.mag_data.append(self.mag / self.N**2)
        self.specific_heat_data.append(self.specific_heat())
        self.susceptibility_data.append(self.susceptibility())
        self.acceptance_rate_data.append(self.accepted_moves / (self.mcs * self.N * self.N))

        # Add average values to storage lists
        self.energy_data_avg.append(self.energy_accum / (self.mcs*self.N**2))
        self.mag_data_avg.append(self.mag_accum / (self.mcs*self.N**2))

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


# Begin by running an Ising MCMC model to equilibrum
model = IsingMCMC(N=32, T=2.5)
model.initialize_random_system()

# Initialize the dashboard
fig, ax = plt.subplots(3, 2)
im = ax[0, 0].imshow(model.lattice, cmap='Greys', vmin=-1, vmax=1)
energy_line, = ax[0, 1].plot([], [], lw=3)
mag_line, = ax[1, 0].plot([], [], lw=3)
heat_line, = ax[1, 1].plot([], [], lw=3)
susceptibility_line, = ax[2, 0].plot([], [], lw=3)
acceptance_line, = ax[2, 1].plot([], [], lw=3)

ax[0, 0].set_title('System')
ax[0, 1].set_ylabel('E / N')
ax[1, 0].set_ylabel('M / N')
ax[1, 1].set_ylabel(r'C_v')
ax[2, 0].set_ylabel(r'$\chi$')
ax[2, 1].set_ylabel('Acceptance Rate')
ax[2, 0].set_xlabel('Monte Carlo Steps')
ax[2, 1].set_xlabel('Monte Carlo Steps')


# Define the animation for the ising/demon model dashboard
def animate_dashboard(i):
    # Run a step in the simulation and display the simulation image
    model.do_one_MC_step()
    im.set_data(model.lattice)

    model.sus
    # Plot temp_data, magnetism, and system energy over monte carlo steps
    x_data = np.arange(model.mcs)
    energy_line.set_data(x_data, model.energy_data)
    mag_line.set_data(x_data, model.mag_data_avg)
    heat_line.set_data(x_data, model.specific_heat_data)
    susceptibility_line.set_data(x_data, model.susceptibility_data)
    acceptance_line.set_data(x_data, model.acceptance_rate_data)

    # Reset the scale and limits of the plots
    # ax[0, 0].relim()
    # ax[0, 0].autoscale_view(True, True, True)
    ax[0, 1].relim()
    ax[0, 1].autoscale_view(True, True, True)
    ax[1, 0].relim()
    ax[1, 0].autoscale_view(True, True, True)
    ax[1, 1].relim()
    ax[1, 1].autoscale_view(True, True, True)
    ax[2, 0].relim()
    ax[2, 0].autoscale_view(True, True, True)
    ax[2, 1].relim()
    ax[2, 1].autoscale_view(True, True, True)


anim = FuncAnimation(fig, animate_dashboard, interval=50, frames=500, repeat=False)
plt.show()
