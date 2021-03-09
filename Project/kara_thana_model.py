"""
Implements the algorithm from the paper:

"A model for predicting forest fire spreading using cellular automata"

https://www.sciencedirect.com/science/article/pii/S0304380096019424
"""
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class KaraThanaFireModel:
    def __init__(self, L, a, init_fire=None):
        self.L = L      # 1D Size of the lattice
        self.a = a      # Size of cell length (m)

        if not init_fire:
            init_fire = [self.L//2, self.L//2]

        # matrix S represents the state of a each cell at time t
        # S[i, j] = A_b / A_t
        # So an unburned cell will have state 0, and a fully burned cell will have state 1.
        self.S = np.zeros([self.L, self.L], dtype=np.float64)

        # Initialize the fire
        self.S[init_fire[0], init_fire[1]] = 0.01

        # matrix R represents a Scalar Velocity field, which is the distribution of the rates of fire spread at every
        # point in a forest
        self.R = np.ones([self.L, self.L], dtype=np.float64)


    def do_one_time_step(self):
        """
        Computes one time step in the CA fire model according to the CA local rule:
        S[i,j] at t + 1 = S[i, j] + (S[i-1, j] + S[i, j-1] + S[i+1, j] + S[i, j+1])
                          + 0.83 (S[i-1, j-1] + S[i-1, j+1] + S[i+1, j-1] + S[i+1, j+1])
        Where all of the LHS S are at time t

        """
        # Compute the CA local rule
        k = [[0.83, 1, 0.83], [1, 1, 1], [0.83, 1, 0.83]]  # kernel for next-nearest neighbor sum
        neighbor_states = signal.convolve2d(self.S, k, mode='same', boundary='fill')

        # If the value of Sij at t+1 is > 1, then set the value equal to 1
        self.S = np.minimum(neighbor_states, 1)

lattice_size = 30   # Number of cells in each dimension
a = 1               # Length of each cell (in meters)
model = KaraThanaFireModel(lattice_size, a)

for i in range(50):
    model.do_one_time_step()
"""
# Define the matplotlib goodies for an animation
frames = 30
fig, ax = plt.subplots(constrained_layout=False)
image = ax.imshow(model.S, cmap='plasma')


# Define a function for each step in the animation
def animate(i):
    image.set_data(model.S)
    model.do_one_time_step()
    ax.set_title(f'fire after {i + 1} iterations')


# Animate iterations of life
anim = FuncAnimation(fig, animate, interval=500, frames=frames, repeat=False)
plt.show()
"""