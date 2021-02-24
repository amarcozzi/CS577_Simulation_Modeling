import numpy as np
from scipy import signal

class ForestFireModel():
    """
    This class implements the Forest-fire model as outlined in Drossel and Schwabl's 1992 paper  "Self-organized
    critical forest-fire model".
    The rules of the cellular automata are as follows:
        1) A burning cell turns into an empty cell
        2) A tree will burn if at least one neighbor is burning
        3) A tree ignites with probability f, even if no neighbor is burning (lightning strike)
        4) An empty space fills with a tree with probabilty p
    """

    def __init__(self, L, p, f, t=.75):
        self.L = L  # Size of the landscape
        self.p = p  # prob of new tree growth
        self.f = f  # prob. of a lightning strike occurring
        self.t = t  # Initial percentage of landscape covered in trees

        # Track burning cells. Initially no cells are burning.
        self.burning = np.zeros([self.L, self.L])

        # Initialize trees on the landscape
        random_trees = np.random.random((self.L, self.L))
        self.trees = np.where(random_trees < self.t, 1, 0)

    def do_one_step(self):
        """ Simulates one time step in the forest-fire system """
        # Step 1) Grow new trees
        self.grow_trees()

        # Step 2) Trees that border neighboring trees catch on fire
        self.grow_fire()

        # Step 3) Ignite trees with a lightning strike

        print('pause')

    def grow_trees(self):
        """ This function grows new trees in empty cells with probability p """
        new_trees = np.random.random((self.L, self.L))
        condition = np.logical_and(self.trees == 0, new_trees < self.p)
        self.trees[condition] = 1

    def grow_fire(self):
        """ If a tree is nearest neighbors with a burning tree, then that tree catches fire """
        # Define kernel for nearest neighbor sum
        k = [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]
             ]
        # Find the nearest neighbors of all of the burning cells
        nn = signal.convolve2d(self.burning, k, mode='same', boundary='wrap')

        # Check if a cell contains a tree, and if it neighbors a burning cell
        condition = np.logical_and(nn > 0, self.trees == 1)

        # Light all of the cells on fire for which the condition is true
        self.burning[condition] = 1

# Create a landscape
landscape = ForestFireModel(5, p=0.1, f=0.001)
landscape.do_one_step()
print('pause')