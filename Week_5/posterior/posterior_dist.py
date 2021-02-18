import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def G(x, m):
    return m[0] * np.exp(m[1] * x) + m[2] * x * np.exp(m[3] * x)


sigma = 0.01
x_data = np.linspace(1, 7, 25)
m_t = np.array([1, -.5, 1, -.75])  # These are the true values that we will attempt to recover

data = G(x_data, m_t) + np.random.randn(x_data.size) * sigma  # The 'data' which we generate with a noise signal

# Initialize our 4 parameters
m_0 = np.zeros(4)
m_0[0] = np.random.uniform(low=0.0, high=2.0)
m_0[1] = np.random.uniform(low=-1.0, high=0.0)
m_0[2] = np.random.uniform(low=0.0, high=2.0)
m_0[3] = np.random.uniform(low=-1.0, high=0.0)

def log_likelihood(d, m, x, s):
    """ Computes the log of the likelihood function """
    return -(1 / 2) * np.sum(np.square(d - G(x, m)) / s ** 2)

def do_mcmc_step(m, d, x, s):
    """ Performs one MCMC step """

    # First we'll generate a proposed solution
    m_p = m + scipy.stats.norm.rvs(size=m.size, scale=s)

    # Find the log of the likelihood function
    log_alpha = np.minimum(0, log_likelihood(d, m_p, x, s) - log_likelihood(d, m, x, s))

    # Generate w in log(rand([0, 1])) and check if w < log(alpha)
    w = np.log(np.random.random())

    if w < log_alpha:   # Accept the proposal
        m_new = m_p
        accept = 1
    else:               # Deny the proposal and keep the old m
        m_new = m
        accept = 0

    return m_new, accept

m_sample = list()
count = 0
solution, accept = do_mcmc_step(m_0, data, x_data, 0.005)
num_accept = accept
acceptance_rate = [accept/1]
for i in range(1, 400000):
    solution, accept = do_mcmc_step(solution, data, x_data, 0.005)
    num_accept += accept
    acceptance_rate.append(num_accept / i)
    # Sample variables every 1000 iterations
    if i % 1000 == 0:
        m_sample.append(solution)

print('done')


plt.plot(x_data, G(x_data, m_t))
# plt.plot(x_data, G(x_data, m_0))
plt.plot(x_data, G(x_data, solution))
plt.legend(['True', 'Sampled'])
plt.show()



# m_sample_array = np.array(m_sample)
#
# plt.plot(range(len(acceptance_rate)), acceptance_rate)
# plt.show()
#
# plt.hist(m_sample_array[:, 0], bins=100)
# plt.show()