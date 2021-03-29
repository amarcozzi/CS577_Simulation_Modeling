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
            Fij = (-p['G'] * p['m'][i] * p['m'][j] / np.power(np.linalg.norm(rij), 3)) * rij

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


""" OLD FORCE FUNCTION
    def force_function(vec, m1, m2, G):
    f = -G * m1 * m2 / np.power(np.linalg.norm(vec), 3)
    return f"""

"""
TEST THE N BODY FUNCTION
"""
"""y_sample = np.array([1.382857, 0,
                     0, 0.157030,
                     -1.382857, 0,
                     0, -0.157030,
                     0, 0.584873,
                     1.871935, 0,
                     0, -0.584873,
                     -1.871935, 0], dtype=np.float128)

euler = np.array([0, 0, 1, 0, -1, 0, 0, 0, 0, .8, 0, -.8])
four_body = np.array([1.382857, 0,
                      0, 0.157030,
                      -1.382857, 0,
                      0, -0.157030,
                      0, 0.584873,
                      1.871935, 0,
                      0, -0.584873,
                      -1.871935, 0], dtype=np.float128)
helium_1 = np.array([0, 0, 2, 0, -1, 0, 0, 0, 0, .95, 0, -1])

p = {'m': [1, 1, 1], 'G': 1, 'dimension': 2, 'fix_first': False}
p4 = {'m': [1, 1, 1, 1], 'G': 1, 'dimension': 2, 'fix_first': False}
phe = {'m': [2, -1, -1], 'G': 1, 'dimension': 2, 'fix_first': True}

headings = ['RUN', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'vx1', 'vy1', 'vx2', 'vy2', 'vx3', 'vy3']
t = PrettyTable(headings)
t.add_row(['euler'] + list(n_body(0, euler, p)))
t.add_row(['He'] + list(n_body(0, helium_1, phe)))
print(t)

headings = ['RUN', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'vx1', 'vy1', 'vx2', 'vy2', 'vx3', 'vy3', 'vx4',
            'vy4']
t = PrettyTable(headings)
t.add_row(['4 body'] + list(n_body(0, four_body, p4)))
print(t)"""

""" ODE LIMITATIONS SECTION """
"""euler = np.array([0, 0, 1, 0, -1, 0, 0, 0, 0, .8, 0, -.8])

montgomery = np.array([0.97000436, -0.24308753, -0.97000436, 0.24308753, 0., 0.,
                       0.466203685, 0.43236573, 0.466203685, 0.43236573,
                       -0.93240737, -0.86473146])
lagrange = np.array([1., 0., -0.5, 0.866025403784439, -0.5, -0.866025403784439,
                     0., 0.8, -0.692820323027551, -0.4, 0.692820323027551, -0.4])

p3 = {'m': [1, 1, 1], 'G': 1, 'dimension': 2, 'fix_first': False}

ode_solver = ODESolver()
tspan = [0, 100]
t, y = ode_solver.solve_ode(n_body, tspan, euler, ode_solver.EulerRichardson, p3, first_step=1)"""

euler      = np.array([0,0,1,0,-1,0,0,0,0,.8,0,-.8])
# p3 = {'m':[1,1,1],'G':1,'dimension':2,'force':gravitational,'fix_first':False}
p3 = {'m':[1,1,1],'G':1,'dimension':2,'fix_first':False}

y0 = euler
p  = p3
d = p['dimension']
dt = 0.0025  # This is wrong - figure it out!
t_span = [0,100]

solver = ODESolver()
t_s,y = solver.solve_ode(n_body,t_span, y0, solver.RK45, p,first_step=dt, atol=1e-6, rtol=1e-6, S=0.98)