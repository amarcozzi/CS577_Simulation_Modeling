"""
Implements the algorithm from the paper:

"A model for predicting forest fire spreading using cellular automata"

https://www.sciencedirect.com/science/article/pii/S0304380096019424
"""

import numpy as np


class CAFireModel:
    def __init__(self, L, a):
        self.L = L      # 1D Size of the lattice
        self.a = a      # Size of cell length (m)

        # matrix R represents a Scalar Velocity field, which is the distribution of the rates of fire spread at every
        # point in a forest
        self.R = np.zeros([self.L, self.L], dtype=np.float64)

        # matrix S represents the state of a each cell at time t
        # S[i, j] = A_b / A_t
        # So an unburned cell will have state 0, and a fully burned cell will have state 1.
        self.S = np.zeros([self.L, self.L], dtype=np.float64)