import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def Euler(y, dt, f, t, *args):
    """ Computes the change in state via the Euler algorithm """
    y_computed = f(t, y, *args) * dt + y
    return y_computed


def EulerCromer(y, dt, f, t, *args):
    """ Computes the change in state via the Euler-Cromer method """
    y_end = Euler(y, dt, f, t, *args)
    return f(t + dt, y_end, *args) * dt + y


def EulerRichardson(y, dt, f, t, *args):
    """ Computes the change in state via the Euler-Richardson method """
    y_mid = Euler(y, dt/2, f, t, *args)
    return f(t + dt / 2, y_mid, *args) * dt + y


def solve_ode(f, tspan, y0, method=Euler, *args, **options):
    """
    Given a function f that returns derivatives,
    dy / dt = f(t, y)
    and an inital state:
    y(tspan[0]) = y0

    This function will return the set of intermediate states of y
    from t0 (tspan[0]) to tf (tspan[1])



    The function is called as follows:

    INPUTS

    f - the function handle to the function that returns derivatives of the
        vector y at time t. The function can also accept parameters that are
        passed via *args, eg f(t,y,g) could accept the acceleration due to gravity.

    tspan - a indexed data type that has [t0 tf] as its two members.
            t0 is the initial time
            tf is the final time

    y0 - The initial state of the system, must be passed as a numpy array.

    method - The method of integrating the ODEs. This week will be one of Euler,
             Euler-Cromer, or Euler-Richardson

    *args - a tuple containing as many additional parameters as you would like for
            the function handle f.

    **options - a dictionary containing all the keywords that might be used to control
                function behavior. For now, there is only one:

                first_step - the initial time step for the simulation.


    OUTPUTS

    t,y

    The returned states will be in the form of a numpy array
    t containing the times the ODEs were solved at and an array
    y with shape tsteps,N_y where tsteps is the number of steps
    and N_y is the number of equations. Observe this makes plotting simple:

    plt.plot(t,y[:,0])

    would plot positions.

    """
    # Add some error handling
    # Pull dt out of options
    dt = options['first_step']
    t0 = tspan[0]
    tf = tspan[1]
    y = [y0]
    t = [t0]
    while t[-1] < tf:
        # Compute change in t and change in position y at each time step
        y.append(method(y[-1], dt, f, t[-1], *args))
        t.append(t[-1] + dt)

    # Convert t, y to np arrays and return them
    return np.array(t), np.array(y)

def simple_gravity(t, y, g):
    """
    This describes the ODEs for the kinematic equations:
    dy/dt =  v
    dv/dt = -g
    """
    result = y[1], -g
    return np.array(result)

g = 9.8
y_0 = np.array([3, 0])
time_interval = (0, np.sqrt(6/g))

from scipy.stats import linregress


def error_scale(steps, errors, plot=True):
    """
    INPUTS -
        steps = a vector of the steps, this will be better if they are logspaced
        errors = a vector errors, usually |y - y_analytic|
        plot = a boolean telling the method to plot, or not
    """
    # Convert delta t steps and errors to log space
    log_steps = np.log10(steps)
    log_errors = np.log10(errors)

    # Run a linear regression error vs. steps
    regression = scipy.stats.linregress(log_steps, log_errors)
    x = np.linspace(np.min(log_steps), np.max(log_steps), 1000)
    y = regression.slope * x + regression.intercept

    if plot:
        plt.plot(log_steps, log_errors, 'k.')
        plt.plot(x, y, 'r-')
        plt.title(f'Slope is {regression.slope:.2f}')
        plt.xlabel('$\log(\Delta t)$')
        plt.ylabel('$\log(error)$')
        plt.show()

    return regression.slope, x, y


def error(y, y_a):
    """
    Simple convenience to compute errors between y and y_a
    the numerical (y) and the analytic solution (y_a)
    """
    # Returns L2 norm of the difference between estimated and analytic solutions
    return np.linalg.norm(y - y_a)


dts = np.logspace(-4, -1, 20)  # You will want these spaced out over several decades

# Initialize storage for the errors at each dt
error_e = list()
error_e_c = list()
error_e_r = list()

# for dt in dts:
#     # Compute analytic solution for dt
#     t = np.arange(time_interval[0], time_interval[1] + dt, dt)
#     y = -(1 / 2) * g * t ** 2 + y_0[1] * t + y_0[0]
#
#     # Call all three different Euler methods and get the results
#     t_e, y_e = solve_ode(simple_gravity, time_interval, y_0, Euler, g, first_step=dt)
#
#     # Compute the errors of all three Euler methods
#     euler_error = error(y_e[:, 0], y)
#
#     # Store the errors for each dt
#     error_e.append(euler_error)

# Regress error on dts for each method
# error_scale(dts, error_e, plot=True)



def sho(t,y,k, m):
    """
    The simple harmonic oscillator
    dy/dt = v
    dv/dt = -k/m y
    """
    velocity = y[1]
    acceleration = -k/m * y[0]
    return np.array([velocity, acceleration])

k = 1
m = 1
y_0 = np.array([0.5, 0])
time_interval = (0, 8*np.pi)
dt = 0.1

# Find analytic solution
t_a = np.arange(0, 8*np.pi + dt, dt)
y_a = y_0[0] * np.cos(np.sqrt(k / m) * t_a) + (y_0[1] / np.sqrt(k / m)) * np.sin(np.sqrt(k / m) * t_a)

t_e, y_e = solve_ode(sho, time_interval, y_0, Euler, k, m, first_step=dt)
# Plot the basic behavior of all 3 algorithms over the time of interest
plt.plot(t_a, y_a[:, 0])
plt.plot(t_e, y_e[:, 0])
