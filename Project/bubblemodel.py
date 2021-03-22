"""
Based on the paper: A hybrid cellular automata/semi-physical model of fire growth
BY: A.L. Sullivan and I.K. Knight

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.6604&rep=rep1&type=pdf
"""

import numpy as np

class Bubble:

    def __init__(self, pos, vertical_velocity):
        self.pos = pos
        self.vertical_velocity = vertical_velocity
        self.number_time_steps = 1

    def update_position(self, dt, wind_vec):
        """
        Moves the bubble up according to its assigned vertical wind speed, and moves the bubble downwind of the wind
        vector
        """
        pass

class BubbleModel:
    def __init__(self, L, V, theta_w):
        self.L = L  # The lattice is size LxL
        # each cell exists in one of three states: unburnt-1, burning-2, burnt-3
        self.world = np.zeros([L, L], dtype=np.int8)
        self.burning_cells = list()

        # Wind parameters
        self.V = V
        self.wind_dir = np.radians(theta_w)

        # Compute wind as a vector and a unit vector
        x_vel = V * np.cos(self.wind_dir)
        y_vel = V * np.sin(self.wind_dir)
        self.wind_vec = np.array([x_vel, y_vel])
        self.unit_wind_vec = self.wind_vec / np.linalg.norm(self.wind_vec)

    def ignite_cell(self, rows, cols):
        """ Assigns the (row, col) pairs to state value 3 to represent ignited fuels """
        self.world[rows, cols] = 2



# Initialize a bubble model
lattice_size = 10
model = BubbleModel(lattice_size, 2, 45)