import numpy as np
import matplotlib.pyplot as plt

from prettytable import PrettyTable
from ode_solver import ODESolver


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

        for i in range(0, N - 1):
            # This loop determines kinetic energy for each body
            ke = p['m'][i] * np.square(np.linalg.norm(vel_matrix[i, :]))
            KE[k] += ke

            # This loop determines total potential energy
            for j in range(i + 1, N):
                rij = pos_matrix[i, :] - pos_matrix[j, :]
                pe = -(p['G'] * p['m'][i] * p['m'][j]) / np.linalg.norm(rij)
                V[k] += pe


    return KE, V, KE + V


euler = np.array([0, 0, 1, 0, -1, 0, 0, 0, 0, .8, 0, -.8])
lagrange = np.array([1., 0., -0.5, 0.866025403784439, -0.5, -0.866025403784439,
                     0., 0.8, -0.692820323027551, -0.4, 0.692820323027551, -0.4])
montgomery = np.array([0.97000436, -0.24308753, -0.97000436, 0.24308753, 0., 0.,
                       0.466203685, 0.43236573, 0.466203685, 0.43236573,
                       -0.93240737, -0.86473146])
p3 = {'m': [1, 1, 1], 'G': 1, 'dimension': 2, 'force': gravitational, 'fix_first': False}

y0 = montgomery
p_init = p3
dt = 0.01  # This is wrong - figure it out!
t_span = [0, 100]

solver = ODESolver()
t_s, y_s = solver.solve_ode(n_body, t_span, y0, solver.RK45, p_init, first_step=dt, atol=1e-10, rtol=1e-14,
                            S=0.98)

kinetic, potential, total = total_energy(y_s, p_init)

plt.plot(kinetic)
plt.plot(potential)
plt.plot(total)
plt.legend(['KE', 'V', 'Total'])
plt.show()
