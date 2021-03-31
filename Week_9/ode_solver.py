import numpy as np


class ODESolver:

    def __init__(self):
        self.num_f_calls = 0
        self.FSAL = 0

    def Euler(self, y, dt, f, t, *args, **options):
        """ Computes the change in state via the Euler algorithm """
        y_computed = f(t, y, *args) * dt + y
        self.num_f_calls += 1  # Increment the number of function calls
        return y_computed, dt

    def EulerCromer(self, y, dt, f, t, *args, **options):
        """ Computes the change in state via the Euler-Cromer method """
        y_end = self.Euler(y, dt, f, t, *args)
        y_return = f(t + dt, y_end, *args) * dt + y
        self.num_f_calls += 1  # Increment the number of function calls
        return y_return, dt

    def EulerRichardson(self, y, dt, f, t, *args, **options):
        """ Computes the change in state via the Euler-Richardson method """
        y_mid = self.Euler(y, dt / 2, f, t, *args)
        y_computed = f(t + dt / 2, y_mid, *args) * dt + y
        self.num_f_calls += 1  # Increment the number of function calls
        return y_computed, dt

    def RungeKutta4(self, y, dt, f, t, *args, **options):
        """ Computes the change in state via the fourth-order Runge-Kutta integration method """
        k1 = dt * f(t, y, *args)
        k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1, *args)
        k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2, *args)
        k4 = dt * f(t + 0.5 * dt, y + k3, *args)

        self.num_f_calls += 4  # Increment the number of function calls
        return y + (1 / 6) * k1 + (1 / 3) * k2 + (1 / 3) * k3 + (1 / 6) * k4, dt

    def RK45(self, y, dt, f, t, *args, **options):
        """ Computes the change in state via the fifth-order Runge-Kutta integration method """
        # Define constant weights
        b = options['b']
        b_star = options['b_star']
        c = options['c']
        c2, c3, c4, c5 = 0.2, 0.3, 0.8, 8.0 / 9.0
        a = options['a']

        k_test = np.zeros([7, y.size])

        """
        while t < t_0 + delta t_0_r:
        for i in range(1, 7):
            yt = y0.copy() + np.dot(a[i], k) * dt
            k[i] = f(t + c[i]*dt, yt, *args)
        y_5 = dot(b, k_6)
        y_4 = dot(b_start
        """

        # compute the k terms for RK5
        k1 = dt * f(t, y, *args)
        k2 = dt * f(t + c2 * dt, y + a[1, 0] * k1, *args)
        k3 = dt * f(t + c3 * dt, y + a[2, 0] * k1 + a[2, 1] * k2, *args)
        k4 = dt * f(t + c4 * dt, y + a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3, *args)
        k5 = dt * f(t + c5 * dt, y + a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4, *args)
        k6 = dt * f(t + dt, y + a[5, 0] * k1 + a[5, 1] * k2 + a[5, 2] * k3 + a[5, 3] * k4 + a[5, 4] * k5, *args)
        k7 = dt * f(t + dt, y + a[6, 0] * k1 + a[6, 1] * k2 + a[6, 2] * k3 + a[6, 3] * k4 + a[6, 4] * k5+ a[6, 5] * k6, *args)
        self.num_f_calls += 6  # Increment the number of function calls
        y_new = y + b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6
        k_array = np.row_stack([k1, k2, k3, k4, k5, k6, k7])
        y_new_test = y + np.dot(b, k_array)

        # compute the error between the 4th and 5th order approximations
        b_diff = b - b_star
        delta = b_diff[0] * k1 + b_diff[1] * k2 + b_diff[2] * k3 + b_diff[3] * k4 + b_diff[4] * k5 + b_diff[5] * k6
        delta_test = np.dot(b - b_star, k_test)
        # This works too
        scale = options['atol'] + options['rtol'] * np.maximum(np.abs(y), np.abs(y_new_test))
        # scale = options['atol'] + options['rtol'] * np.abs(y_new)
        # err = np.sqrt(np.mean((delta / scale) ** 2))
        err = np.sqrt(np.mean((delta_test / scale) ** 2))

        # Compute the new time step, h_new, based on errors between 4th and 5th order Runge-Kutta methods
        # dt_new = options['S'] * dt * np.power(1 / err, 1 / 5)
        dt_new = options['S'] * dt * np.power(1 / err, 1 / 5)

        # Limit the up and down scaling of the time step
        if dt_new > dt:
            dt_new = min(dt_new, dt * 10)
        else:
            dt_new = max(dt_new, dt / 5)

        # reject the step and try again with a smaller dt
        if err > 1:
            y_new, dt_new = self.RK45(y, dt_new, f, t, *args, **options)

        return y_new, dt_new

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
        # Reset the number of function calls
        self.num_f_calls = 0
        self.FSAL = 0

        # pull in solver parameters from passed arguments
        dt = options['first_step']
        t0 = tspan[0]
        tf = tspan[1]
        y = [y0]
        t = [t0]

        # Check if the method call is RK45, if so, pack the options with the Dormand-Prince coefficients
        if method == self.RK45:
            options['b'] = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
            options['b_star'] = np.array(
                [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
            options['c'] = np.array([1, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
            options['a'] = np.array([[0, 0, 0, 0, 0, 0],
                                     [1 / 5, 0, 0, 0, 0, 0],
                                     [3 / 40, 9 / 40, 0, 0, 0, 0],
                                     [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
                                     [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
                                     [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
                                     [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]])

        while t[-1] < tf:
            current_t = t[-1]
            # Compute change in t and change in position y at each time step
            """if method == self.RK45:
                new_y, dt = method(y[-1], dt, f, t[-1], *args, **options)
            else:
                new_y = method(y[-1], dt, f, t[-1], *args, **options)"""
            new_y, dt = method(y[-1], dt, f, t[-1], *args, **options)
            y.append(new_y)
            t.append(t[-1] + dt)

        print(f'The force function was called {self.num_f_calls} times.')
        # Convert t, y to np arrays and return them
        return np.array(t), np.array(y)
