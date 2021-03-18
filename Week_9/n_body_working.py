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
    # TODO: multiply m1 and m2
    # TODO: fix_first

    # Get some constants and initialize
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
            Fij = p['force'](rij, p['mass'][i], p['mass'][j], p['G']) * rij

            # Fill in the symmetric pieces of the force matrix
            Fmatrix[i, j, :] = Fij / p['mass'][i]
            Fmatrix[j, i, :] = -Fij / p['mass'][j]

    # Compute the force vectors acting on each body
    forces = np.sum(Fmatrix, axis=1)

    # flatten the forces matrix and combine it with the vector list
    acc_vec = forces.flatten().tolist()
    dxdt = np.array([vel_vec, acc_vec]).flatten()

    if p['fix_first']:
        # Find indices that need to be zero
        dxdt[:d] = 0
        dxdt[half:half+d] = 0

    return dxdt

def force_function(vec, m1, m2, G):
    """ Computes the force from the n-body problem """
    f = -G * m1 * m2 / np.power(np.linalg.norm(vec), 3)
    return f

class Bodies:
    """ Object representing properties of the bodies in the n-body problem """
    def __init__(self, n, d, m, g, fix_first=False):
        self.num_bodies = n
        self.dim = d
        self.G = g
        self.fix_first = fix_first

        # Check that the number of masses is equal to the number of bodies
        if len(m) != n:
            raise ValueError('Number of masses must equal the number of bodies')
        else:
            self.masses = m


################################
### TEST THE N BODY FUNCTION ###
################################
"""y_sample = np.array([1.382857, 0,
                     0, 0.157030,
                     -1.382857, 0,
                     0, -0.157030,
                     0, 0.584873,
                     1.871935, 0,
                     0, -0.584873,
                     -1.871935, 0], dtype=np.float128)

euler      = np.array([0,0,1,0,-1,0,0,0,0,.8,0,-.8])
four_body  = np.array([1.382857,0,\
                   0,0.157030,\
                  -1.382857,0,\
                   0,-0.157030,\
                   0,0.584873,\
                   1.871935,0,\
                   0,-0.584873,\
                  -1.871935,0],dtype=np.float128)
helium_1 = np.array([0,0,2,0,-1,0,0,0,0,.95,0,-1])

p = {'num_bodies': 3,'mass':[1,1,1],'G':1,'dimension':2, 'force':force_function,'fix_first':False}
p4 = {'num_bodies': 4, 'mass':[1,1,1,1],'G':1,'dimension':2,'force':force_function, 'fix_first':False}
phe = {'num_bodies': 3, 'mass':[2,-1,-1],'G':1,'dimension':2,'fix_first':True,'force':force_function}



headings = ['RUN','x1','y1','x2','y2','x3','y3','vx1','vy1','vx2','vy2','vx3','vy3']
t = PrettyTable(headings)
t.add_row(['euler']+list(n_body(0,euler,p)))
t.add_row(['He']+list(n_body(0,helium_1,phe)))
print(t)

headings = ['RUN','x1','y1','x2','y2','x3','y3','x4','y4','vx1','vy1','vx2','vy2','vx3','vy3','vx4','vy4']
t = PrettyTable(headings)
t.add_row(['4 body']+list(n_body(0,four_body,p4)))
print(t)"""

euler      = np.array([0,0,1,0,-1,0,0,0,0,.8,0,-.8])

montgomery = np.array([0.97000436,-0.24308753,-0.97000436,0.24308753, 0., 0.,\
                    0.466203685, 0.43236573, 0.466203685, 0.43236573,\
                   -0.93240737,-0.86473146])
lagrange   = np.array([1.,0.,-0.5,0.866025403784439, -0.5,-0.866025403784439,\
                  0.,0.8,-0.692820323027551,-0.4, 0.692820323027551, -0.4])

p3 = {'num_bodies': 3, 'mass':[1,1,1],'G':1,'dimension':2,'force':force_function,'fix_first':False}

ode_solver = ODESolver()
tspan = [0, 100]
t, y = ode_solver.solve_ode(n_body, tspan, euler, ode_solver.EulerRichardson, p3, first_step=1)
print('debug')