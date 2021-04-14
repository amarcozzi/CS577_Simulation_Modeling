import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial
from scipy import integrate
from scipy import optimize


class SEIRModel:

    def __init__(self, p_data_file, d_data_file):
        # Store the population and deaths data
        self.pdata, self.ddata = self._parse_data(p_data_file, d_data_file)

        # Other attributes to initialized in class methods
        self.loc_deaths = None
        self.dates = []
        self.p = dict()
        self.y_init = list()
        self.data_weekly_deaths = None
        self.model_weekly_deaths = None
        self.Ro = 0
        self.solution = None
        self.params_list = []
        self.days_from_zero = []
        self.iters=0

    @staticmethod
    def _parse_data(pop, deaths):
        """ Takes in two csv files and parses the deaths and population data """
        # Read in population data
        pop_data = pd.read_csv(pop)
        pop_data = pop_data[["NAME", "POPESTIMATE2019"]]

        # Read in weekly deaths data
        deaths_data = pd.read_csv(deaths, date_parser=True)
        deaths_data = deaths_data[["End Date", "State", "COVID-19 Deaths", "Group"]]  # Read in just these columns
        deaths_data = deaths_data.loc[deaths_data['Group'] == "By Week"]  # Filter by weekly data
        deaths_data['COVID-19 Deaths'] = deaths_data['COVID-19 Deaths'].fillna(0)  # Replace nan with 0
        deaths_data['End Date'] = pd.to_datetime(deaths_data['End Date'])  # Convert to dates

        return pop_data, deaths_data

    def SEIR(self, t, y):
        """
        Accepts the state of the system as:
        y[0] = Susceptible population - never impacted by the disease
        y[1] = Exposed population - has the disease, but not yet manifesting symptoms
        y[2] = Infected population - has the disease, and symptoms
        y[3] = Recovered population - has recovered from the disease, and is no longer susceptible
        y[4] = Dead population - these are the people that have died from COVID-19

        This function returns the time derivative of the state, dydt, and uses a dictionary
        of parameters p.
        """
        # Pull parameters out of state vector and parameter dictionary
        S, E, I, R, D = y[:]
        beta, delta, gamma, q, d, N = self.p['beta'], self.p['delta'], self.p['gamma'], self.p['q'], self.p['d'], \
                                      self.p['N']

        # Compute terms in the system of ODE. Do this now since we repeat computations
        beta_eval = np.polynomial.polynomial.polyval(t, beta)
        S_change = beta_eval * S * (I + q * E) / N
        E_change = E / delta
        I_change = I / gamma

        # Compute derivatives with terms above
        dS = -S_change
        dE = S_change - E_change
        dI = E_change - I_change
        dR = (1 - d) * I_change
        dD = d * I_change

        return np.array([dS, dE, dI, dR, dD])

    def set_parameters(self, q=0.5, delta=6, gamma=10, death_rate=0.01, Eo_frac=0.00001, coefs=None):
        """
        This simple routine simply sets the parameters for the model.
        Note they are not all unity, I want you to figure out the
        appropriate parameter values.
        location - location to be modeled
        q - the attenuation of the infectivity, gamma in the population that is E
        delta - the length of time of asymptomatic exposure
        gamma - the length of time of infection
        death_rate - the fraction of infected dying
        Eo_frac    - a small number that is multiplied by the population to give the initially exposed
        degree     - degree of polynomial used for beta(t)
        coeffs     - the set of initial coefficients for the polynomial, None in many cases
        beta_o     - a constant initial value of beta, will be changed in the optimization
        """
        # Set beta as a polynomial object
        # beta = np.polynomial.Polynomial(coefs)
        self.p['beta'] = coefs

        # Fill out the rest of the parameters from function arguments
        self.p['q'] = q
        self.p['delta'] = delta
        self.p['gamma'] = gamma
        self.p['d'] = death_rate

        # Fill out the state vector y
        S = self.p['N']
        E = S * Eo_frac
        I = 0
        R = 0
        D = 0
        self.y_init = np.array([S, E, I, R, D])

    def set_location(self, location):
        """
        Given a location string, this function will read appropriate
        data files to set the population parameter N and
        data fields within the SEIR object appropriate time series of
        deaths from COVID-19.
        """
        # Get the population of the location
        loc_subframe = self.pdata.query(f"NAME == '{location}'")
        loc_pop = loc_subframe.POPESTIMATE2019.values[0]
        self.p['N'] = loc_pop

        # Get the death date from the location
        self.loc_deaths = self.ddata.query(f"State == '{location}'")
        self.data_weekly_deaths = self.loc_deaths['COVID-19 Deaths'].to_numpy()

        # Get a list of dates to compare the model with data
        self.dates = self.loc_deaths['End Date'].tolist()
        self.days_from_zero = []
        for day in self.dates:
            self.days_from_zero.append((day - self.dates[0]).days)

    def get_SSE(self, opt_params, *args):
        """
        The hardest working routine - will
        1. accept a set of parameters for the polynomial coefficients and the death rate
        2. run the SEIR model using the ODE solver, solve_ivp
        3. return an Sum Square Error by comparing model result to data.
        """
        self.set_parameters(q=args[0], delta=args[1], gamma=args[2], death_rate=opt_params[-1], Eo_frac=args[3],
                            coefs=opt_params[:-1])
        self.solution = integrate.solve_ivp(self.SEIR, (0, self.days_from_zero[-1]), self.y_init, 'RK45',
                                            self.days_from_zero, atol=1e-6, rtol=1e-6)
        self.model_weekly_deaths = self._convert_cum_to_weekly()

        # Compute Ro of the solution
        # TODO: This will change when beta is a function of time
        self.Ro = np.polynomial.polynomial.polyval(self.days_from_zero[-1], self.p['beta']) * self.p['gamma'] * \
                  np.ones(self.model_weekly_deaths.size)

        # Compute the error of the computation
        if self.data_weekly_deaths.shape == self.model_weekly_deaths.shape:
            square_error = np.square(self.data_weekly_deaths - self.model_weekly_deaths)
            sse_iter = np.sum(square_error)
        else:
            sse_iter = np.inf

        self.iters += 1
        print(f'iteration {self.iters} has sse of: {sse_iter:20.2f} with params {opt_params}')
        return sse_iter

    def optimize_model(self, opt_params: tuple, fixed_params: tuple, method: str = 'nelder-mead', kwargs=None):
        """ Runs optimization on the SEIR model """
        x0 = np.array(opt_params)
        result = optimize.minimize(self.get_SSE, x0, fixed_params, method,
                                   options={'xatol': kwargs['xatol'], 'disp': kwargs['disp']},
                                   callback=self._store_location)
        return result

    def plot_results(self):
        """
        create a 4 panel plot with the following views:

        * Deaths modeled and deaths observed as a function of time.
        * 𝑅𝑜  as a function of time.
        * The susceptible, infected and recovered populations as a function of time.
        * The fraction of the population that has recovered as a function time.
        Observe that by passing a list of dates you can get a nicely formatted time axis.
        """
        # Initialize the dashboard
        fig = plt.figure(figsize=(20, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        # Plot the lines
        model_deaths_line, = ax1.plot(self.dates, self.model_weekly_deaths, lw=3, c='k')
        data_deaths_line, = ax1.plot(self.dates, self.data_weekly_deaths, lw=3, c='r')
        ro_line, = ax2.plot(self.dates, self.Ro, lw=3, c='r')
        S_line, = ax3.plot(self.dates, self.solution.y[0, :], c='k')
        I_line, = ax3.plot(self.dates, self.solution.y[2, :], c='y')
        R_line, = ax3.plot(self.dates, self.solution.y[3, :], c='g')
        R_per_pop_line, = ax4.plot(self.dates, self.solution.y[3, :] / self.p['N'], c='g')
        herd_immunity_line, = ax4.plot(self.dates, 1 - (1 / self.Ro), c='k')

        # Set the titles, legends, axis, etc.
        ax1.set_title('Deaths')
        ax1.set_ylabel('Deaths')
        ax1.legend(['Modeled', 'Observed'])
        ax2.set_title(r'$R_o$')
        ax2.set_ylabel('Additional cases per case')
        ax3.set_ylabel('Persons')
        ax3.legend(['Susceptible', 'Infected', 'Recovered'])
        ax4.set_ylabel('Percent of Population')
        ax4.legend(['Natural Immunity', 'Herd Immunity'])

        plt.show()

    def _store_location(self, x):
        self.params_list.append(x)

    def _convert_cum_to_weekly(self):
        """ Converts the cumulative weekly model output to cumulative totals """
        cum_deaths = self.solution.y.T[:, 4]
        weekly_deaths = np.zeros(cum_deaths.shape)
        for i in range(1, cum_deaths.shape[0]):
            weekly_deaths[i] = cum_deaths[i] - cum_deaths[i - 1]
        return weekly_deaths


# Load in the data and initialize the class instance
population_data_file = './data/nst-est2019-alldata.csv'
deaths_data_file = './data/Provisional_COVID-19_Death_Counts_by_Week_Ending_Date_and_State.csv'
seir = SEIRModel(population_data_file, deaths_data_file)
seir.set_location('New York')

# Pack the Parameters
#                   b0  b1 b2 b3 b4 b5 b6    d
params_optimize = (0.095, 0, 0, 0, 0, 0, 0, 0.02)
#               q   delta gamma    E0
params_fixed = (0.5, 6, 10, 0.00001)
options = {'xatol': 1e-2, 'disp': True}

# Run the optimization
res = seir.optimize_model(params_optimize, params_fixed, method='nelder-mead', kwargs=options)
seir.plot_results()