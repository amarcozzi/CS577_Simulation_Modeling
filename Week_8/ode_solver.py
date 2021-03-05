import numpy as np

class ODESolver:

    def Euler(self, y, dt, f, t, *args):
        """ Computes the change in state via the Euler algorithm """
        y_computed = f(t, y, *args) * dt + y
        return y_computed

    def EulerCromer(self, y, dt, f, t, *args):
        """ Computes the change in state via the Euler-Cromer method """
        y_end = self.Euler(y, dt, f, t, *args)
        return f(t + dt, y_end, *args) * dt + y

    def EulerRichardson(self, y, dt, f, t, *args):
        """ Computes the change in state via the Euler-Richardson method """
        y_mid = self.Euler(y, dt / 2, f, t, *args)
        return f(t + dt / 2, y_mid, *args) * dt + y

    def solve_ode(self, f, tspan, y0, method=Euler, *args, **options):
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