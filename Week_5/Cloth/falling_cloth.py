import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Cloth:
    def __init__(self, N=16, kT=1e-3):
        # Initialize constants/input parameters
        self.N = N  # Number of masses per side.
        self.k = 50  # Spring constant
        self.length = 1  # Spring equilibrium length
        self.z = 1  # starting z position of the cloth
        self.m = .04  # Mass, almost looks like a density in the limit of N -> inf
        self.sigma = .25 * self.length  # This is standard deviation of the gaussian region sampled for perturbations.
        self.g = 9.8  # gravitational acceleration, note sign!
        self.kT = kT  # Temperature appearing in Boltzmann factor, this should be changed through runs.
        self.E = 0  # Initial energy
        self.step = 0  # Starting mc steps
        self.E_data = list()  # Track energy over time
        self.E_data.append(self.E)  # Add starting energy to data

        # Initialize lattice. Lattice is a dictionary with coordinate tuple (i, j) as the key
        # with 3D coordinates np array [x, y, z] as the position.
        self.lattice = dict()
        for i in range(0, self.N):
            for j in range(0, self.N):
                self.lattice[(i, j)] = np.array([i * self.length, j * self.length, self.z], dtype=np.float64)

        # Want to add a set of points to be stationary. This would be equivalent to either holding the cloth
        # at the corners, pinching the cloth somewhere, or dropping the cloth onto an object


    def do_mcmc_step(self):
        for i in range(self.N ** 2):
            dE, pt, dr = self.perturb()
            if dE < 0 or np.random.rand() < np.exp(-dE / self.kT):
                self.lattice[pt] += dr
                self.E += dE
        self.E_data.append(self.E)
        self.step += 1

    def perturb(self):
        """ Picks a random, non-corner point, and perturbs that point in the x, y, or z direction """
        pt = self._pick_random_point()

        # pick random direction / quantity to perturb
        dx, dy, dz = 0, 0, 0
        p = np.random.rand()
        if p < .25:
            dx = np.random.randn() * self.sigma
        elif p < .5:
            dy = np.random.randn() * self.sigma
        else:
            dz = np.random.randn() * 2 * self.sigma

        # change in position due to the perturbing
        dr = np.array([dx, dy, dz])

        # Compute the change in energy as a result of perturbation
        dE = self.calc_dE(pt, dr)

        return dE, pt, dr

    def calc_dE(self, pt, dr):
        """ Computes the change in energy from perturbation """
        # prep lists, write new function to get nn
        nn = self._get_nn(pt)

        # prep vectors by pulling them out of self.cloth
        r = self.lattice[pt]
        rp = r + dr

        # sum in delta E
        energy_sum = 0
        for n in nn:
            nr = self.lattice[n]    # position of nearest neighbor
            dp = np.linalg.norm(rp - nr)    # distance between r' and nearest neighbor
            d = np.linalg.norm(r - nr)      # distance between r and nearest neighbor
            energy = np.square(dp - self.length) - np.square(d - self.length)
            energy_sum += energy
            pass

        dE = (1 / 2) * self.k * energy_sum + (self.m * self.g * dr[2])

        return dE

    def get_world_coords(self):
        """ Plots the cloth """
        x = list()
        y = list()
        z = list()
        world_coords = list(self.lattice.values())
        for i in range(len(world_coords)):
            x.append(world_coords[i][0])
            y.append(world_coords[i][1])
            z.append(world_coords[i][2])

        return np.array(x), np.array(y), np.array(z)

    def _get_nn(self, pt):
        """ Returns a list of nearest neighbors of point """
        nn = list()
        left = (pt[0] - 1, pt[1])
        right = (pt[0] + 1, pt[1])
        up = (pt[0], pt[1] - 1)
        down = (pt[0], pt[1] + 1)
        if left in self.lattice:
            nn.append(left)
        if right in self.lattice:
            nn.append(right)
        if up in self.lattice:
            nn.append(up)
        if down in self.lattice:
            nn.append(down)

        return nn

    def _pick_random_point(self):
        """ Returns a random point on the lattice """
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        while self._is_corner(i, j):
            i = np.random.randint(0, self.N)
            j = np.random.randint(0, self.N)

        return i, j

    def _is_corner(self, i, j):
        """ Returns true if pt is a corner point, false otherwise """
        if i == 0 and j == 0:
            return True
        elif i == 0 and j == self.N - 1:
            return True
        elif i == self.N - 1 and j == 0:
            return True
        elif i == self.N - 1 and j == self.N - 1:
            return True
        else:
            return False


cloth = Cloth(N=64)
X, Y, Z = cloth.get_world_coords()

for k in range(100):
    cloth.do_mcmc_step()
    pass


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = cloth.get_world_coords()
plot3d = ax.plot_trisurf(X, Y, Z)
# plot3d = ax.scatter(X, Y, Z)
plt.show()
