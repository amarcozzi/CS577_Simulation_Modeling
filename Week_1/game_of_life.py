import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal


def game_of_life(A, bc='Periodic'):
    """
    Given a matrix A of 0s and 1s, this function will apply the rules of the "Game of Life" to the matrix A.

    :param np.ndarray A: 2D Matrix of "life" to apply the rule of the game of life on. Must consist of 0s and 1s.
    :param str bc: Boundary condition on the matrix. Supports 'Periodic', 'Symmetric', or 'Open'.
        Sets a periodic boundary condition by default.
    :return: Matrix A after one iteration of the game of life.
    :rtype: np.ndarray
    """
    # Perform Nearest Neighbor interpolation to get sum of next nearest neighbors for every "cell" in A
    # Perform next nearest neighbor with a periodic boundary condition
    # A_NN_slow = nearest_neighbor_interp_slow(A)
    A_NN = nearest_neighbor_interp_fast(A, bc)

    # Apply the rules of the game of life to the matrix A. Find whether a live cell stays alive, or a life-less cell
    # finds life. Et en A[rcadia] ergo.
    remains_alive_condition = np.logical_and(A == 1, np.logical_or(A_NN == 2, A_NN == 3))
    reproduction_condition = np.logical_and(A == 0, A_NN == 3)

    # Apply the remains alive conditions and reproduction conditions. If either is true, populate the cell with a 1,
    # if neither is true, make the cell a 0.
    A_new = np.where(np.logical_or(remains_alive_condition, reproduction_condition), 1, 0).astype(np.int8)

    return A_new


def nearest_neighbor_interp_fast(A, bc):
    """
    This function uses the convolve2d function from scipy.signal to find the sum of the next nearest neighbors around
    every cell of A.

    Note: I read about how to use convolution2d and apply it to the sum of the next-nearest neighbors problem from
    the scipy documentation page, and the following stack overflow thread:
    https://stackoverflow.com/questions/35925317/vectorizing-sum-of-nearest-neighbours-in-matrix-using-numpy
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

    This approach is significantly faster than the "slow" version I wrote in nearest_neighbor_interp_slow which uses
    for loops to iterate over the elements of the matrix, and conditionals to check for boundary conditions. For small
    matrices, like 3x3 or 5x5, the difference is negligible, but for large matrices the difference is significant. For
    example, for a 1000x1000 matrix the "slow" approach takes 67,0494 ms, whereas the scipy convolve2d approach
    takes 393 ms. Pretty significant...

    :param np.ndarray A: 2D Matrix representing life
    :return: Sum of next nearest neighbors for every cell in A
    :rtype: np.ndarray
    """
    # Define kernel for next-nearest neighbor sum
    k = [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]
         ]
    k_big = [[True, True, True, True, True],
             [True, True, True, True, True],
             [True, True, False, True, True],
             [True, True, True, True, True],
             [True, True, True, True, True]
             ]

    # Enforce periodic boundary conditions on the convolution
    if bc == 'Periodic':
        convolution = signal.convolve2d(A, k, mode='same', boundary='wrap')
    # Enforce symmetric boundary conditions on the convolution
    elif bc == 'Symmetric':
        convolution = signal.convolve2d(A, k, mode='same', boundary='symm')
    # Enforce open boundary conditions on the convolution
    else:
        convolution = signal.convolve2d(A, k, mode='same', boundary='fill')

    return convolution


def nearest_neighbor_interp_slow(A):
    """
    This function performs the next nearest neighbor interpolation on each cell in a matrix A, and returns the total
    count of next nearest neighbors in that cell.

    :param np.ndarray A: Matrix to interpolate.
    :return: Matrix whose cells are equal to the total count of next nearest neighbors in the corresponding cell in A.
    :rtype: np.ndarray
    """
    # Get the dimensions of the array
    m, n = A.shape

    # Initialize matrix to hold the sum of the nearest neighbors
    nn = np.zeros([m, n])

    # Loop through the cells of the array
    for j in range(m):
        for i in range(n):
            # Need to get the value of the cells to the left, right, up, down, and diagonal of cell i, j.
            # AND need to check the boundary conditions for each of those cells.
            j_left, i_left = periodic_boundary(m, n, j, i - 1)
            j_right, i_right = periodic_boundary(m, n, j, i + 1)
            j_up, i_up = periodic_boundary(m, n, j - 1, i)
            j_down, i_down = periodic_boundary(m, n, j + 1, i)
            j_top_left, i_top_left = periodic_boundary(m, n, j - 1, i - 1)
            j_top_right, i_top_right = periodic_boundary(m, n, j - 1, i + 1)
            j_down_left, i_down_left = periodic_boundary(m, n, j + 1, i - 1)
            j_down_right, i_down_right = periodic_boundary(m, n, j + 1, i + 1)

            # Find the value of the cell in each neighbor around i, j
            left = A[j_left, i_left]
            right = A[j_right, i_right]
            up = A[j_up, i_up]
            down = A[j_down, i_down]
            top_left = A[j_top_left, i_top_left]
            top_right = A[j_top_right, i_top_right]
            down_left = A[j_down_left, i_down_left]
            down_right = A[j_down_right, i_down_right]

            nn[j, i] = left + right + up + down + top_left + top_right + down_left + down_right

    return nn


def periodic_boundary(rows, cols, j, i):
    """
    Computes the corrected indices for a periodic boundary on a matrix of size m, n. e.g. If this function receives the
    j, i coordinates (3, 1) for a 3x3 matrix, then a periodic boundary condition would correct that coordinate to
    (0, 1)

    :param int rows: number of rows in the matrix
    :param int cols: number of cols in the matrix
    :param int j: row index
    :param int i: column index
    :return: (j, i) The corrected row and column indices
    :rtype: tuple
    """
    # Check if the B.C. is violated on the top left corner
    if j < 0 and i < 0:
        # Send j, i to the bottom right corner of the matrix
        j = rows - 1
        i = cols - 1

        # Check if the B.C. is violated on the top right corner
    elif j < 0 and i == cols:
        # Send j, i to the bottom left corner of the matrix
        j = rows - 1
        i = 0

    # Check if the B.C. is violated on the bottom left corner
    elif j > rows - 1 and i < 0:
        # Send j, i to the top right corner of the matrix
        j = 0
        i = cols - 1

    # Check if the B.C. is violated on the bottom right corner
    elif j > rows - 1 and i > cols - 1:
        # Send j, i to the top left corner of the matrix
        j = 0
        i = 0

        # Check if the B.C. is violated to the left
    elif i < 0:
        # Send the cell to the right side of the matrix
        i = cols - 1

    # Check if the B.C. is violated to the right
    elif i > cols - 1:
        # Send the cell to the right side of the matrix
        i = 0

    # Check if the B.C. is violated above
    elif j < 0:
        # Send the cell to the bottom of the matrix
        j = rows - 1

    # Check if the B.C. is violated below
    elif j > rows - 1:
        # Send the cell to the top of the matrix
        j = 0

    return j, i


def initialize_random_life(size, p=.5):
    """
    Takes in a tuple of size and returns an np array with life randomly generated
    :param tuple size: Size of the life environment
    :param float p: Probability that life begins in a cell
    :return: Initial conditions of the life simulation
    :rtype: tuple
    """
    A = np.random.random(size)
    return np.where(A > p, 0, 1).astype(np.int8)


# Initialize the life environment randomly
initial_prob = 0.5
life = initialize_random_life((100, 100), initial_prob)


# Define the matplotlib goodies for an animation
frames = 1000
fig, ax = plt.subplots(constrained_layout=False)
image = ax.imshow(life)


# Define a function for each step in the animation
def animate(i):
    global life
    image.set_data(life)
    life = game_of_life(life, bc='Periodic')
    biodensity = np.mean(life) * 100
    ax.set_title(f'Life after {i + 1} iterations.\nEnvironment is {biodensity:.2f}% full of life.')


# Animate iterations of life
anim = FuncAnimation(fig, animate, interval=100, frames=frames, repeat=False)
plt.show()
# anim.save('game_of_life_animation.mp4')

# How many times should we iterate life, and how many different initial probabilities
# should we consider?
iters = 1000
probs = 4
p = 0.5

# Initialize matrix to hold values of rho
rho = np.zeros([iters, 2, probs])

p_count = 0
# Iterate over 5 different probabilities of initial life on the board
for p in np.linspace(0.2, .8, probs):
    life = initialize_random_life((32, 32), p)
    rho[0, 0, p_count] = np.mean(life)
    for i in range(1, iters):
        life = game_of_life(life)
        density = np.mean(life)
        rho[i, 0, p_count] = density
        rho[i-1, 1, p_count] = density
    p_count += 1

fig1, ax = plt.subplots(2, 2)
ax[0, 0].plot(rho[:-1, 0, 0], rho[:-1, 1, 0], '.b')
ax[0, 0].set_title('Density at p_init = 0.2')
ax[0, 1].plot(rho[:-1, 0, 1], rho[:-1, 1, 1], '.b')
ax[0, 1].set_title('Density at p_init = 0.4')
ax[1, 0].plot(rho[:-1, 0, 2], rho[:-1, 1, 2], '.b')
ax[1, 0].set_title('Density at p_init = 0.6')
ax[1, 1].plot(rho[:-1, 0, 3], rho[:-1, 1, 3], '.b')
ax[1, 1].set_title('Density at p_init = 0.8')
plt.show()

