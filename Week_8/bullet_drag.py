import numpy as np
import matplotlib.pyplot as plt
from ode_solver import ODESolver
from scipy import interpolate

# Load in the data for the reference coefficients of drag for G1 and G7 bullets
g1_drag_data = np.genfromtxt('./g1_drag_data.txt')
g7_drag_data = np.genfromtxt('./g7_drag_data.txt')

# Convert from K_d to C_d?
g1_drag_data[:, 1] *= np.pi/4
g7_drag_data[:, 1] *= np.pi/4

# Plot the G1 and G7 data
# plt.subplots(figsize=(10, 4.5))
# plt.plot(g1_drag_data[:, 0], g1_drag_data[:, 1], c='C4')
# plt.plot(g7_drag_data[:, 0], g7_drag_data[:, 1], c='C6')
# plt.vlines(1.0, ymin=0, ymax=1)
# plt.legend(['G1', 'G7'])
# plt.xlabel(r'Mach Number, $m$')
# plt.ylabel(r'$C_d$')
# plt.grid()
# plt.show()


# Define the derivative solver.
def projectile(t, y, b):
    """
    Takes in a position and velocity vector, returns the derivative, a velocity and acceleration vector.
    In this cas we have position, velocity = (x, y, vx, vy)
    and velocity, acceleration = (vx, vy, ax, ay)

    t is not used
    y is the input state
    b is an object containing the ballistic model
    returns dy/dt
    """
    # x and y velocities
    vx = y[2]
    vy = y[3]

    # v magnitude
    v_mag = np.sqrt(vx ** 2 + vy ** 2)

    # Drag force
    Fd = b.get_drag(v_mag)

    # x and y acceleration
    ax = -Fd * (vx / v_mag)  # The vx/v_mag is the Fdx component of Fd
    ay = -b.g - Fd * (vy / v_mag)  # The vy/v_mag is the Fdy component of Fd

    return np.array([vx, vy, ax, ay])


class BallisticCoefficient:
    """ Object represents the aerodynamic properties of a user-designated caliber bullet """

    def __init__(self, caliber, bc, rho, vs, drag_ref_data, units='imperial'):
        """

        :param caliber: Caliber of the projectile
        :param bc: ballistic coefficient of the projectile
        :param units: metric or imperial
        :param drag_ref_data
        """
        self.caliber = caliber
        self.bc = bc
        self.rho = rho
        self.vs = vs
        self.units = units
        self.drag_ref_data = drag_ref_data

        if units == 'imperial':
            self.g = 32.2
        if units == 'metric':
            self.g = 9.8

    def get_drag(self, v):
        """
        Takes in a velocity magnitude. Returns the magnitude of the drag resistance.

        :param v:
        :return:
        """
        # Assumes imperial units for now

        # Compute mach number and reference coefficient of drag
        vm = v / self.vs
        Cdg = np.interp(vm, self.drag_ref_data[:, 0], self.drag_ref_data[:, 1])

        force_of_drag = (1 / 2) * (1 / self.bc) * self.rho * Cdg * np.square(v)
        return force_of_drag

        # if self.units == 'imperial':
        #     pass
        #
        # if self.units == 'metric':
        #     pass


# Do a sample simulation with the 6.5mm Creedmoor 144gr Long Range Hybrid Target
# Has G1 BC: 0.655
# And G7 BC: 0.336
# with muzzle velocity 2830 ft/s
# https://bergerbullets.com/information/lines-and-designs/long-range-hybrid-target-bullets/

# Load in observed data for the above round
lapua_6_5_creedmoor_144gr = np.genfromtxt('./lapua_6_5_creedmoor_144gr_data.txt')
lapua_6_5_creedmoor_144gr[:, 0] *= 3    # convert range from yards to feet
lapua_6_5_creedmoor_144gr[:, 2] *= 12   # Convert drop from inches to feet

# Assume temp celsius = 20
air_density = 0.07517           # lb/ft^3
speed_of_sound = 1126           # ft/s
ballistics = BallisticCoefficient('6.5 Creedmoor', bc=0.655, rho=air_density, vs=speed_of_sound,
                                 drag_ref_data=g1_drag_data, units='imperial')


range = np.array([0, 1000])                 # x range in feet
v_muzzle = 2830                             # ft/s
dist_between_barrel_and_scope = -0.164      # in feet
zero_range = 600                            # in feet
theta = np.arctan(dist_between_barrel_and_scope / zero_range)   # angle above horizontal to hit the zero range
vx_0 = v_muzzle * np.cos(theta)     # Initial x velocity
vy_0 = v_muzzle * np.sin(theta)     # Initial y velocity
initial_state = np.array([0.0, dist_between_barrel_and_scope, vx_0, vy_0])   # position in inches, velocity in ft/s

ode_solver = ODESolver()
t, y = ode_solver.solve_ode(projectile, range, initial_state, ode_solver.EulerRichardson, ballistics, first_step=0.01)

# find index where y[0] >= 3000
far_range = np.argwhere(y[:, 0] >= 3000)

plt.subplots(figsize=(10, 4.5))
plt.plot(y[:, far_range], y[:, 1])
plt.plot(lapua_6_5_creedmoor_144gr[:, 0], lapua_6_5_creedmoor_144gr[:, 2])
plt.grid()
plt.show()
print('nada')