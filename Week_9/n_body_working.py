import numpy as np


def n_body(t, y, p):
    """
    Write what goes in here!
    Instructions above.

    t is our time vector
    y is the state vector of all the bodies:
    [x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, etc...]
    p_init is the object with our data

    note:

    returns dxdt
    """

    # Get some constants and initialize
    N = len(p['m'])
    d = p['dimension']
    Fmatrix = np.zeros([N, N, d])

    # split y into position and velocity vectors, go from flattened pos. vector to size Nxd array
    half = y.size // 2
    pos_vec = y[:half]
    vel_vec = y[half:]
    pos_matrix = pos_vec.reshape(N, d)

    # Loop over the top right corner of the force matrix
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            # Find the vector from body j to body i
            rij = pos_matrix[i, :] - pos_matrix[j, :]

            # Compute the force from body j onto body i
            Fij = p['force'](p['G'], p['m'][i], p['m'][j], rij)

            # Fill in the symmetric pieces of the force matrix
            Fmatrix[i, j, :] = Fij / p['m'][i]
            Fmatrix[j, i, :] = -Fij / p['m'][j]

    # Compute the force vectors acting on each body
    forces = np.sum(Fmatrix, axis=1)

    # flatten the forces matrix and combine it with the vector list
    # TODO: Rework how you build these
    acc_vec = forces.flatten().tolist()
    dxdt = np.array([vel_vec, acc_vec]).flatten()

    if p['fix_first']:
        # Find indices that need to be zero
        dxdt[:d] = 0
        dxdt[half:half + d] = 0

    return dxdt


def gravitational(G, m1, m2, r):
    """ Computes the gravitational force between two bodies with large mass """
    return (-G * m1 * m2 / np.power(np.linalg.norm(r), 3)) * r


def total_energy(y, p):
    steps, dofs = y.shape
    d = p['dimension']
    N = len(p['m'])
    half = dofs // 2

    # KE = V = np.zeros(steps)
    KE = np.zeros(steps)
    V = np.zeros(steps)
    for k in range(steps):
        y_k = y[k, :]
        pos_vec = y_k[:half]
        vel_vec = y_k[half:]
        pos_matrix = pos_vec.reshape(N, d)
        vel_matrix = vel_vec.reshape(N, d)

        # This loop determines kinetic energy for each body
        for i in range(0, N):
            ke = (1 / 2) * p['m'][i] * np.square(np.linalg.norm(vel_matrix[i, :]))
            KE[k] += ke

        # This loop determines total potential energy
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                rij = pos_matrix[i, :] - pos_matrix[j, :]
                pe = -(p['G'] * p['m'][i] * p['m'][j]) / np.linalg.norm(rij)
                V[k] += pe

    return KE, V, KE + V

def subtract_velocity_center_of_mass(p, y):

    m = p['m']
    n = len(m)
    d = p['dimension']

    half = y.size // 2
    vel_vec = y[half:]
    vel_matrix = vel_vec.reshape(n, d)

    upper_x = 0
    upper_y = 0
    lower = 0
    for i in range(n):
        upper_x += m[i] * vel_matrix[i, 0]
        upper_y += m[i] * vel_matrix[i, 1]
        lower += m[i]
    com = np.row_stack([upper_x / lower, upper_y / lower])

    vel_vec[::2] -= com[0]
    vel_vec[1::2] -= com[1]
    y[half:] = vel_vec

    return y

