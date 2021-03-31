import numpy as np

from prettytable import PrettyTable
from ode_solver import ODESolver


def n_body(t, y, p):
    """
    Write what goes in here!
    Instructions above.

    t is our time vector
    y is the state vector of all the bodies:
    [x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, etc...]
    p is the object with our data

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


euler      = np.array([0,0,1,0,-1,0,0,0,0,.8,0,-.8])
p3 = {'m':[1,1,1],'G':1,'dimension':2,'force':gravitational,'fix_first':False}

y0 = euler
p  = p3
d = p['dimension']
dt = 0.0025  # This is wrong - figure it out!
t_span = [0,100]

solver = ODESolver()
t_s,y = solver.solve_ode(n_body,t_span, y0, solver.RK45, p,first_step=dt, atol=1e-6, rtol=1e-6, S=0.9)