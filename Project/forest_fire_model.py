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
        self.burning = np.zeros([self.L, self.L], dtype=np.int8)

        # Initialize trees on the landscape
        random_trees = np.random.random((self.L, self.L))
        self.trees = np.where(random_trees < self.t, 1, 0).astype(np.int8)

        # Initialize tree and fire counts
        self.num_trees = [np.count_nonzero(self.trees == 1)]
        self.num_fire = [0]

        self.iters = 0
        self.time_steps = [0]

    def do_one_step(self):
        """ Simulates one time step in the forest-fire system """
        # Step 0) Preprocess tree/fire count lists for the new iteration
        self.num_trees.append(0)
        self.num_fire.append(0)

        # Step 1) Grow new trees
        self.grow_trees()

        # Step 2) Trees that border neighboring trees catch on fire
        self.grow_fire()

        # Step 3) Ignite trees with a lightning strike
        self.throw_lightning()

        self.iters += 1
        self.time_steps.append(self.iters)

    def grow_trees(self):
        """ This function grows new trees in empty cells with probability p_init """
        new_trees = np.random.random((self.L, self.L))
        condition = np.logical_and(self.trees == 0, new_trees < self.p)
        self.trees[condition] = 1

        self.num_trees[-1] += self.num_trees[-2] + np.count_nonzero(condition)

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
        condition = np.logical_and(nn > 0, self.trees == 1).astype(bool)

        # Light all of the cells on fire for which the condition is true
        self.burning[condition] = 1

        # Extinguish the previously burning cells and remove their trees
        self.burning[prev_burn] = 0
        self.trees[prev_burn] = 0

        # Remove trees that are now on fire, increment number of burning cells
        num_converted_cells = np.count_nonzero(condition)
        self.num_trees[-1] -= num_converted_cells
        self.num_fire[-1] += num_converted_cells

    def throw_lightning(self):
        """ Randomly ignites a tree with probability f """
        lightning_prob = np.random.random((self.L, self.L))
        strikes = np.where(lightning_prob < self.f, True, False)
        condition = np.logical_and(strikes == 1, self.trees == 1).astype(bool)

        # Set locations where lightning strikes AND there is a tree to burning
        self.burning[condition] = 1

        # Remove trees that are now on fire, increment number of burning cells
        num_converted_cells = np.count_nonzero(condition)
        self.num_trees[-1] -= num_converted_cells
        self.num_fire[-1] += num_converted_cells

    def animate_forest(self, frames, figsize):
        """ Runs an instance of the forest with an animated plot """
        # Create custom colormaps for burning and for vegetation
        # burn_cmap = LinearSegmentedColormap.from_list('burn_cmap', ['firebrick'], N=1)
        veg_cmap = LinearSegmentedColormap.from_list('veg_cmap', ['black', 'forestgreen'], N=2)

        # Define the matplotlib goodies for an animation
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
        veg_image = ax.imshow(self.trees, cmap=veg_cmap)
        burn_data = np.ma.masked_where(self.burning < 0.05, self.burning)
        burn_image = ax.imshow(burn_data, cmap='Spectral')

        # Define a function for each step in the animation
        def animate(i):
            self.do_one_step()
            veg_image.set_data(self.trees)
            burn_data = np.ma.masked_where(self.burning < 0.05, self.burning)
            burn_image.set_data(burn_data)

        # Animate iterations of life
        anim = FuncAnimation(fig, animate, interval=100, frames=frames, repeat=False)
        plt.close()

        return anim

    def plot_stats(self, figsize, normalized=False):
        """ Plots the number of trees and fire cells over the course of the simulation """
        if normalized:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(self.num_fire, self.num_trees)
            """trees_norm = np.array([self.num_trees]) / self.L**2
            fire_norm = np.array([self.num_fire]) / self.L**2

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(self.time_steps, trees_norm.ravel(), c='tab:green')
            ax.plot(self.time_steps, fire_norm.ravel(), c='tab:red')"""

        else:

            figure, ax1 = plt.subplots(figsize=figsize)
            ax1.plot(self.time_steps, self.num_trees, c='tab:green')
            ax1.set_xlabel('Iterations (steps)')
            ax1.set_ylabel('Number of Trees')
            # ax1.tick_params(axis='y', labelcolor='tab:green')
            ax1.tick_params(axis='y')

            ax2 = ax1.twinx()
            ax2.plot(self.time_steps, self.num_fire, c='tab:red')
            ax2.set_ylabel('Number of Fire Cells')
            # ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.tick_params(axis='y')

            figure.tight_layout()
            figure.legend(['Tree Cells', 'Fire Cells'], loc="upper right", bbox_to_anchor=(1, 1),
                          bbox_transform=ax1.transAxes)
        plt.show()


"""# Create a landscape
landscape = ForestFireModel(250, p=0.002, f=0.00001, t=.5)

# Create custom colormaps for burning and for vegetation
# burn_cmap = LinearSegmentedColormap.from_list('burn_cmap', ['firebrick'], N=1)
veg_cmap = LinearSegmentedColormap.from_list('veg_cmap', ['black', 'forestgreen'], N=2)

# Define the matplotlib goodies for an animation
frames = 10000
fig, ax = plt.subplots(constrained_layout=False)
veg_image = ax.imshow(landscape.trees, cmap=veg_cmap, interpolation='nearest')
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
plt.show()"""
