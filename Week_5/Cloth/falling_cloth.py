import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Cloth:
    def __init__(self, static_pts, N=16, kT=1e-3, thresh=0.01, cooling_steps=3):
        # Initialize constants/input parameters
        self.static_pts = static_pts
        self.thresh = thresh
        self.end_flag = False
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
        self.E_change = list()

        # Set up cooling schedule
        cooling_s

        # Initialize lattice. Lattice is a dictionary with coordinate tuple (i, j) as the key
        # with 3D coordinates np array [x, y, z] as the position.
        self.lattice = dict()
        for i in range(0, self.N):
            for j in range(0, self.N):
                self.lattice[(i, j)] = np.array([i * self.length, j * self.length, self.z], dtype=np.float64)

        # Want to add a set of points to be stationary. This would be equivalent to either holding the cloth
        # at the corners, pinching the cloth somewhere, or dropping the cloth onto an object

    def simulate(self):
        """ Runs the simulation until the threshold condition is met """
        while not self.end_flag:
            self.do_mcmc_step()
        print(f'Simulation finished after {self.step} MCMC steps')

    def do_mcmc_step(self):
        for i in range(self.N ** 2):
            dE, pt, dr = self.perturb()
            if dE < 0 or np.random.rand() < np.exp(-dE / self.kT):
                self.lattice[pt] += dr
                self.E += dE
        self.step += 1
        # Record E in list
        self.E_data.append(self.E)
        if self.step > 1:
            # compute percentage change in E
            change = (self.E_data[self.step] - self.E_data[self.step - 1]) / self.E_data[self.step - 1]
            self.E_change.append(change)
            if self.step > 5:
                # Stop the simulation if the threshold is reached
                test_value = abs(sum(self.E_change[-5:-1]))
                if test_value < self.thresh:
                    self.end_flag = True
                else:
                    pass

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
            nr = self.lattice[n]  # position of nearest neighbor
            dp = np.linalg.norm(rp - nr)  # distance between r' and nearest neighbor
            d = np.linalg.norm(r - nr)  # distance between r and nearest neighbor
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
        while self._is_static((i, j)):
            i = np.random.randint(0, self.N)
            j = np.random.randint(0, self.N)

        return i, j

    def _is_static(self, pt):
        """ Returns true if pt is in the set of static points, false otherwise """
        if pt in self.static_pts:
            return True
        else:
            return False


L = 16
corners = {(0, 0): None,
           (0, L - 1): None,
           (L - 1, 0): None,
           (L - 1, L - 1): None}
cloth = Cloth(corners, N=L, thresh=0.01, cooling_steps=3)
cloth.simulate()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = cloth.get_world_coords()
wf_plot = ax.plot_wireframe(X.reshape(cloth.N, cloth.N).T, Y.reshape(cloth.N, cloth.N).T, Z.reshape(cloth.N, cloth.N).T)
plt.show()
