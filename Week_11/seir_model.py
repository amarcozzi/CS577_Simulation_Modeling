import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

def SEIR(t, y, p):
    """
    Accepts the state of the system as:
    y[0] = Suseptable population - never impacted by the disease
    y[1] = Exposed population - has the disease, but not yet manifesting symptoms
    y[2] = Infected population - has the disease, and symptoms
    y[3] = Recovered population - has recovered from the disease, and is no longer suseptable

    This function returns the time deriviative of the state, dydt, and uses a dictionary
    of parameters p.
    """
    # Pull parameters out of state vector and parameter dictionary
    S, E, I, R = y[:]
    beta, delta, gamma, q, N = p['beta'], p['delta'], p['gamma'], p['q'], p['N']

    # Compute terms in the system of ODE. Do this now since we repeat computations
    S_change = beta * S * (I + q * E)
    E_change = E / delta
    I_change = I / gamma

    # Compute derivatives with terms above
    dS = -S_change / N
    dE = S_change - E_change
    dI = E_change - I_change
    dR = I_change

    return np.row_stack([dS, dE, dI, dR])


def set_parameter(self, location="", q=1, delta=1, gamma=1,
                  death_rate=1, Eo_frac=1, degree=1,
                  coefs=1, beta_o=1):
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
    beta = np.polynomial.Polynomial([0.08, 0, 0, 0, 0, 0])
    print('debug')


def set_location(self, location):
    """
    Given a location string, this function will read appropriate
    data files to set the population parameter N and
    data fields within the SEIR object appropriate time series of
    deaths from COVID-19.
    """
    pass


def get_SSE(self, opt_params):
    """
    The hardest working routine - will
    1. accept a set of parameters for the polynomial coefficients and the death rate
    2. run the SEIR model using the ODE solver, solve_ivp
    3. return an Sum Square Error by comparing model result to data.
    """


def plot_results(self):
    """
    create a 4 panel plot with the following views:

    * Deaths modeled and deaths observed as a function of time.
    * 𝑅𝑜  as a function of time.
    * The susceptable, infected and recovered populations as a function of time.
    * The fraction of the population that has recovered as a function time.
    Observe that by passing a list of dates you can get a nicely formated time axis.
    """

def load_data(f):
    """ Takes in a data file, and then outputs pandas data file """
    d = pd.read_csv(f, delimiter=',', parse_dates=True)
    return d



file = 'Provisional_COVID-19_Death_Counts_by_Week_Ending_Date_and_State.csv'
data = pd.read_csv(file)
print('debug')
