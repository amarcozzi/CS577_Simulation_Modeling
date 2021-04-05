import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


class ForestFireModel:
    """
    This class implements the Forest-fire model as outlined in Drossel and Schwabl's 1992 paper  "Self-organized
    critical forest-fire model".
    The rules of the cellular automata are as follows:
        1) A burning cell turns into an empty cell
        2) A tree will burn if at least one neighbor is burning
        3) A tree ignites with probability f, even if no neighbor is burning (lightning strike)
        4) An empty space fills with a tree with probability p_init
    """

    def __init__(self, L, p, f, t=.75):
        self.L = L  # Size of the landscape
        self.p = p  # prob of new tree growth
        self.f = f  # prob. of a lightning strike occurring
        self.t = t  # Initial percentage of landscape covered in trees

        # Track burning cells. Initially no cells are burning.
        self.burning = np.zeros([self.L, self.L], dtype=np.int)
        # DEBUG: Set one area to burning
        self.burning[2, 2] = 1

        # Initialize trees on the landscape
        random_trees = np.random.random((self.L, self.L))
        self.trees = np.where(random_trees < self.t, 1, 0).astype(np.int)
        # DEBUG make sure there's a tree in burning area
        self.trees[2, 2] = 1

    def do_one_step(self):
        """ Simulates one time step in the forest-fire system """
        # Step 1) Grow new trees
        self.grow_trees()

        # Step 2) Trees that border neighboring trees catch on fire
        self.grow_fire()

        # Step 3) Ignite trees with a lightning strike
        self.throw_lightning()

    def grow_trees(self):
        """ This function grows new trees in empty cells with probability p_init """
        new_trees = np.random.random((self.L, self.L))
        condition = np.logical_and(self.trees == 0, new_trees < self.p)
        self.trees[condition] = 1

    def grow_fire(self):
        """ If a tree is nearest neighbors with a burning tree, then that tree catches fire """

        # Find all of the previously burning sites
        prev_burn = np.where(self.burning == 1, True, False)

        # Find the nearest neighbors of all of the burning cells
        k = [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]
             ]
        nn = signal.convolve2d(self.burning, k, mode='same', boundary='wrap')

        # Check if a cell contains a tree, and if it neighbors a burning cell
        condition = np.logical_and(nn > 0, self.trees == 1).astype(np.bool)

        # Light all of the cells on fire for which the condition is true
        self.burning[condition] = 1

        # Extinguish the previously burning cells and remove their trees
        self.burning[prev_burn] = 0
        self.trees[prev_burn] = 0

    def throw_lightning(self):
        """ Randomly ignites a tree with probability f """
        lightning_prob = np.random.random((self.L, self.L))
        strikes = np.where(lightning_prob < self.f, True, False)
        condition = np.logical_and(strikes == 1, self.trees == 1).astype(np.bool)

        # Set locations where lightning strikes AND there is a tree to burning
        self.burning[condition] = 1


# Create a landscape
landscape = ForestFireModel(100, p=0.002, f=0.00001, t=.5)

# Create custom colormaps for burning and for vegetation
# burn_cmap = LinearSegmentedColormap.from_list('burn_cmap', ['firebrick'], N=1)
veg_cmap = LinearSegmentedColormap.from_list('veg_cmap', ['black', 'forestgreen'], N=2)

# Define the matplotlib goodies for an animation
frames = 10000
fig, ax = plt.subplots(constrained_layout=False)
veg_image = ax.imshow(landscape.trees, cmap=veg_cmap)
burn_data = np.ma.masked_where(landscape.burning < 0.05, landscape.burning)
burn_image = ax.imshow(burn_data, cmap='Spectral')


# Define a function for each step in the animation
def animate(i):
    landscape.do_one_step()
    veg_image.set_data(landscape.trees)
    burn_data = np.ma.masked_where(landscape.burning < 0.05, landscape.burning)
    burn_image.set_data(burn_data)


# Animate iterations of life
anim = FuncAnimation(fig, animate, interval=100, frames=frames, repeat=False)
plt.show()
