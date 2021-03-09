"""
Implements the Cellular Automata wildfire model from the following paper:

A cellular automata model for forest fire spread prediction: The caseof the wildfire that swept through Spetses Island
in 1990
A. Alexandridisa, D. Vakalisb, C.I. Siettosc, G.V. Bafasa
https://www.sciencedirect.com/science/article/pii/S0096300308004943
"""
import numpy as np
from scipy import signal


class GreekModel:
    """
    Object represents the Cellular Automata presented in the paper
    """
    def __init__(self, L):
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

        # Wind speed constants
        V = 8  # wind speed m/s
        theta_w = np.radians(90)  # Wind direction (Degrees from 0)
        ft = np.exp(V * c2 * (np.cos(theta_w) - 1))
        p_w = np.exp(c1 * V) * ft

        # Slope constants
        theta_s = 0
        p_s = np.exp(a * theta_s)

        # Compute the probability of burning
        self.p_burn = ph * (1 + p_veg) * (1 + p_den) * p_w * p_s

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
        no_fuel = np.where(self.State == 1, 1, 0)
        consumed = np.where(self.State == 4, 4, 0)
        recently_burned = np.where(self.State == 3, 4, 0)
        actively_burning = np.where(self.State == 3, 1, 0)

        # Now the complicated one, rule 4
        k = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]  # kernel for next-nearest neighbor sum
        potential_states = signal.convolve2d(actively_burning, k, mode='same', boundary='fill')
        chance_to_burn = np.random.rand(self.L, self.L)
        successfull_transfer = np.logical_and(potential_states == 1, chance_to_burn < self.p_burn)
        successfull_transfer = np.logical_and(successfull_transfer, self.State == 2)
        newly_burning = np.where(successfull_transfer, 3, 0)

        # Reconstruct the state matrix
        self.State = no_fuel + recently_burned + consumed + newly_burning


# Initialize the lattice
lattice_size = 5
model = GreekModel(lattice_size)

# Initial fuel/no fuel in the lattice
fuel_probability = 1.0
model.initialize_state(fuel_probability)

# Set an ignition point
ignition_point_r = [lattice_size//2]
ignition_point_c = [lattice_size//2]
model.ignite_fire(ignition_point_r, ignition_point_c)

for i in range(50):
    model.do_one_time_step()
    pass