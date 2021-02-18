import numpy as np
import matplotlib.pyplot as plt


def Euler(u, dt, f, t, *args):
    """ Computes the change in state via the Euler algorithm """
    return f(t, u, *args) * dt + u


def EulerCromer():
    pass


def EulerRichardson():
    pass


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
        # Keep track of t
        # Initialize y = y0
        y.append(method(y[-1], dt, f, t[-1], *args))  # Accounts for a changing dt
        t.append(t[-1] + dt)

    # Convert t, y to np arrays and return them
    return np.array(t), np.array(y)


def simple_gravity(t, y, g):
    """
    This describes the ODEs for the kinematic equations:
    dy/dt =  v
    dv/dt = -g
    """
    result = y[1] * t, -g * t
    return np.array(result)

g = 9.8
y_0 = np.array([3, 0])
t_e, y_e = solve_ode(simple_gravity, (0, np.sqrt(6/g)), y_0, Euler, g, first_step=0.1)
plt.plot(t_e, y_e[:, 0])
plt.show()


