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

    def __init__(self, L, V, theta):
        """
        Initializes the State of the system according to the CA definition:
        1) State = 1: Cell contains no fuel
        2) State = 2: Cell contains unburned fuel
        3) State = 3: Cell contains actively burning fuel
        4) State = 4: Cell has no more combustible fuel left
        """
        self.L = L  # Size of the lattice
        self.State = np.zeros([self.L, self.L])

        # Add some of the paper constants. To be fixed up later
        p_den = 0
        p_veg = 0.4
        ph = 0.58
        a = 0.078
        c1 = 0.045
        c2 = 0.131

        """ # Wind speed constants
        V = 2  # wind speed m/s
        theta_w = np.radians(180)  # Wind direction (Degrees from 0)
        ft = np.exp(V * c2 * (np.cos(theta_w) - 1))
        p_w = np.exp(c1 * V) * ft"""
        # p_w = self._compute_wind_probs(theta, V, c1, c2)

        # Slope constants
        theta_s = 0
        p_s = np.exp(a * theta_s)

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

        # Compute the probability that a state neighboring a burning state will burn
        k = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]  # kernel for next-nearest neighbor sum
        potential_states = signal.convolve2d(prev_burning, k, mode='same', boundary='fill')

        theta_vec = self._get_fire_propagation_vectors()

        chance_to_burn = np.random.rand(self.L, self.L)

        # Need to border a currently burning state, follow some probability, and have available fuel
        condition_one = np.logical_and(potential_states == 1, chance_to_burn < self.p_burn)
        condition_two = np.logical_and(condition_one, self.State == 2)
        newly_burning = np.where(condition_two, True, False)

        # Reconstruct the state matrix
        self.State[prev_burning] = 4  # Set previously burning cells to extinguished
        self.State[newly_burning] = 3  # Set new cells on fire

    def _compute_wind_probs(self, wind_dir, V, c1, c2):
        """

        """
        # Compute wind as a vector
        wind_dir_rads = np.radians(wind_dir)    # Convert from degrees to radians
        x_vel = V * np.sin(wind_dir_rads)
        y_vel = V * np.cos(wind_dir_rads)
        wind_vec = np.array([x_vel, y_vel])

        # Compute theta for all cells
        theta = self._compute_wind_thetas(wind_vec)

        ft = np.exp(V * c2 * (np.cos(theta) - 1))
        p_w = np.exp(c1 * V) * ft

        return p_w

    def _compute_wind_thetas(self, wind_dir_vec):
        """
        Computes the angle between all vectors on the discretized plane and the vector representing the wind direction
        """

        unit_wind_dir_vec = wind_dir_vec / np.linalg.norm(wind_dir_vec)
        angle_matrix = np.zeros([self.L, self.L])

        x = np.linspace(-self.L//2, self.L//2, self.L)
        y = np.linspace(-self.L//2, self.L//2, self.L)
        xx, yy = np.meshgrid(x, y)

        for i in range(self.L):
            for j in range(self.L):
                x_coord = xx[i, j]
                y_coord = yy[i, j]
                rij = np.array([x_coord, y_coord])
                unit_rij = rij / np.linalg.norm(rij)
                dot_product = np.dot(unit_rij, unit_wind_dir_vec)
                angle = np.arccos(dot_product)
                angle_matrix[i, j] = angle
        return angle_matrix

    def _get_fire_propagation_vectors(self, m):
        """
        Computes the angle between the wind vector and the direction of fire propagation for every cell
        """

# Initialize the lattice
lattice_size = 1000
model = GreekModel(lattice_size, 8, 45)

# Initial fuel/no fuel in the lattice
fuel_probability = 1.0
model.initialize_state(fuel_probability)

# Set an ignition point
ignition_point_r = [lattice_size // 2]
ignition_point_c = [lattice_size // 2]
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
