import numpy as np


def n_body(t, y, p):
    '''
    Write what goes in here!
    Instructions above.

    t is our time vector
    y is the state vector of all the bodies:
    [x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, etc...]
    p is the object with our data

    note:

    returns dxdt
    '''
    N = p['num_bodies']
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
            Fij = p['force'](rij, p['mass'][j], p['G']) * rij

            # Fill in the symmetric pieces of the force matrix
            Fmatrix[i, j, :] = Fij
            Fmatrix[j, i, :] = -Fij

    # Compute the force vectors acting on each body
    forces = np.sum(Fmatrix, axis=1)

    # flatten the forces matrix and combine it with the vector list
    acc_vec = forces.flatten().tolist()
    dxdt = vel_vec + acc_vec

    return dxdt


def force_function(vec, m, G):
    """ Computes the force from the n-body problem """
    f = -G * m / np.power(np.linalg.norm(vec), 3)
    return f


bodies = {'num_bodies': 3,
          'dimension': 2,
          'G': 1,
          'mass': [1000, 5000, 1500],
          'force': force_function}

y_sample = np.array([1.382857, 0,
                     0, 0.157030,
                     -1.382857, 0,
                     0, 0.584873,
                     1.871935, 0,
                     0, -0.584873, ], dtype=np.float128)

t_0 = 0
test = n_body(t_0, y_sample, bodies)