"""
Implements the Cellular Automata wildfire model from the following paper:

A cellular automata model for forest fire spread prediction: The caseof the wildfire that swept through Spetses Island
in 1990
A. Alexandridisa, D. Vakalisb, C.I. Siettosc, G.V. Bafasa
https://www.sciencedirect.com/science/article/pii/S0096300308004943
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap


class GreekModel:
    """
    Object represents the Cellular Automata presented in the paper
    """

    def __init__(self, L, V, wind_dir):
        """

        Parameters:
            L - Size of the lattice
            V - Wind speed
            wind_dir - Wind direction. Measured in degrees from true north, clockwise

        Initializes the State of the system according to the CA definition:
        1) State = 1: Cell contains no fuel
        2) State = 2: Cell contains unburned fuel
        3) State = 3: Cell contains actively burning fuel
        4) State = 4: Cell has no more combustible fuel left
        """
        self.L = L  # Size of the lattice
        self.State = np.zeros([self.L, self.L])

        # Add some of the paper constants. To be fixed up later
        self.p_den = 0
        self.p_veg = 0.4
        self.ph = 0.58
        self.a = 0.078
        self.c1 = 0.045
        self.c2 = 0.131

        """ # Wind speed constants
        V = 2  # wind speed m/s
        theta_w = np.radians(180)  # Wind direction (Degrees from 0)
        ft = np.exp(V * c2 * (np.cos(theta_w) - 1))
        p_w = np.exp(c1 * V) * ft"""
        # p_w = self._compute_wind_probs(theta, V, c1, c2)
        self.V = V
        self.wind_dir = np.radians(wind_dir)

        # Compute wind as a vector
        x_vel = V * np.sin(self.wind_dir)
        y_vel = V * np.cos(self.wind_dir)
        self.wind_vec = np.array([x_vel, y_vel])
        self.unit_wind_vec = self.wind_vec / np.linalg.norm(self.wind_vec)
        self.kx, self.ky = self._get_wind_kernels()

        # Slope constants
        # theta_s = 0
        # p_s = np.exp(a * theta_s)

        # Compute the probability of burning
        # self.p_burn = ph * (1 + p_veg) * (1 + p_den) * p_w * p_s
        print('debug')

    def initialize_state(self, p):
        """ Initializes the state to contain either burned or unburned """
        random_fuels = np.random.rand(self.L, self.L)
        self.State = np.where(random_fuels < p, 2, 1)

    def ignite_fire(self, rows, cols):
        """ Assigns the (row, col) pairs to state value 3 to represent ignited fuels """
        self.State[rows, cols] = 3

    def do_one_time_step(self):
        """
        Performs one time step according to the following CA rules:
        1) If state(i,j,t) = 1 -> state(i,j,t+1) = 1
           Fuel that cannot burn will not burn
        2) If state(i,j,t) = 3 -> state(i,j,t+1) = 4
           Fuel that burned in the last time step will consume in the next time step
        3) If state(i,j,t) = 4 -> state(i,j,t+1) = 4
           Fuel will not reburn
        4) If state(i,j,t) = 3 -> state(i+-1,j+-1,t+1) = 3 with probability p_burn
        """
        # Break down the State matrix into its components
        prev_burning = np.where(self.State == 3, True, False)

        # Compute the number of neighbors of each cell
        k = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]  # kernel for next-nearest neighbor sum
        potential_states = signal.convolve2d(prev_burning, k, mode='same', boundary='fill')

        # Compute the angles of fire propagation relative to the wind vector
        theta_w = self._get_fire_propagation_vectors(prev_burning)

        # Compute the probability of burning for each cell
        p_burn = self._get_fire_probability(theta_w)

        random_matrix = np.random.rand(self.L, self.L)

        # Need to border a currently burning state, follow some probability, and have available fuel
        condition_one = np.logical_and(potential_states >= 1, random_matrix < p_burn)
        condition_two = np.logical_and(condition_one, self.State == 2)
        newly_burning = np.where(condition_two, True, False)

        # Reconstruct the state matrix
        self.State[prev_burning] = 4  # Set previously burning cells to extinguished
        self.State[newly_burning] = 3  # Set new cells on fire


    def _get_wind_kernels(self):
        """ Computes the x and y direction convolution kernels based on wind direction """
        # Create 3x3 grid for the kernel
        x = np.linspace(-1, 1, 3)
        y = np.linspace(-1, 1, 3)
        xx, yy = np.meshgrid(x, y)

        # Compute projection of vector ij onto the wind vector
        k = np.zeros([3, 3])
        angles = np.zeros([3, 3])
        for i in range(3):
            for j in range(3):
                x_coord = xx[i, j]
                y_coord = yy[i, j]
                rij = np.array([x_coord, y_coord])
                dot_product = np.dot(rij, self.unit_wind_vec)
                if dot_product < 0:
                    dot_product += 1e-8
                k[j, i] = dot_product

        left_x = k[1, 0] + np.flip(k.T)[1, 0]
        right_x = k[1, 2] + np.flip(k.T)[1, 2]
        top_y = k[0, 1] + np.flip(k.T)[0, 1]
        bottom_y = k[2, 1] + np.flip(k.T)[2, 1]

        kx = np.array([[left_x, 0, right_x], [left_x, 0, right_x], [left_x, 0, right_x]])
        ky = np.array([[top_y, top_y, top_y], [0, 0, 0], [bottom_y, bottom_y, bottom_y]])

        return kx, ky

    def _get_fire_propagation_vectors(self, m):
        """
        Computes the angle between the wind vector and the direction of fire propagation for every cell
        """
        x = signal.convolve2d(m, self.kx, mode='same', boundary='fill')
        y = signal.convolve2d(m, self.ky, mode='same', boundary='fill')

        # Compute unit vectors of x and y components
        xy_vec = np.column_stack([y.flatten(), x.flatten()])
        xy_unit_vec = np.divide(xy_vec, np.linalg.norm(xy_vec, axis=1)[:, None])
        # np.nan_to_num(xy_unit_vec, copy=False, nan=0.0)

        # Take the dot product between unit vectors and find the angle between them
        dot_product = np.dot(xy_unit_vec, self.unit_wind_vec)
        angle = np.arccos(dot_product)
        theta_w_matrix = angle
        view_theta = np.degrees(theta_w_matrix).reshape(self.L, self.L)

        return theta_w_matrix.reshape(self.L, self.L)

    def _get_fire_probability(self, t_w):
        """ Computes the probability of burning for each cell """
        # Compute probability of wind carried fire propagation
        ft = np.exp(self.V * self.c2 * (np.cos(t_w) - 1))
        p_w = np.exp(self.c1 * self.V) * ft

        # Compute probability of slope carried fire propagation
        theta_s = 0
        p_s = np.exp(self.a * theta_s)

        # Compute the probability of burning
        # p_b = self.ph * (1 + self.p_veg) * (1 + self.p_den) * p_w * p_s
        p_b = self.ph * (0.5 + self.p_veg) * (1 + self.p_den) * p_w * p_s

        return p_b

# Initialize the lattice
lattice_size = 1000
model = GreekModel(lattice_size, 8, 0)

# Initial fuel/no fuel in the lattice
fuel_probability = 1.0
model.initialize_state(fuel_probability)

# Set an ignition point
ignition_point_r = [50]
ignition_point_c = [30]
model.ignite_fire(ignition_point_r, ignition_point_c)

"""for i in range(50):
    model.do_one_time_step()
    pass"""

# Create custom colormaps for burning and for vegetation
color_dict = {1: "bisque",
              2: "forestgreen",
              3: "red",
              4: "dimgrey"}

# We create a colormar from our list of colors
cm = ListedColormap([color_dict[x] for x in color_dict.keys()])

# Let's also define the description of each category
labels = np.array(["No Fuel", "Available Fuel", "Burning", "Consumed"])

# Prepare bins for the normalizer
norm_bins = np.sort([*color_dict.keys()]) + 0.5
norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

# Make normalizer and formatter
norm = mpl.colors.BoundaryNorm(norm_bins, len(labels), clip=True)
fmt = mpl.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

# Define the matplotlib goodies for an animation
frames = 10000
fig, ax = plt.subplots()
world = ax.imshow(model.State, cmap=cm, norm=norm)
diff = norm_bins[1:] - norm_bins[:-1]
tickz = norm_bins[:-1] + diff / 2
cb = fig.colorbar(world, format=fmt, ticks=tickz)


# Define a function for each step in the animation
def animate(i):
    model.do_one_time_step()
    world.set_data(model.State)


# Animate the fire model
anim = FuncAnimation(fig, animate, interval=50, frames=frames, repeat=False)
plt.show()
