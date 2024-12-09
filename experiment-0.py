# This script performs a sanity check. In case x_{0} ~ N(mu_{0}, sigma^{2}_{0}), 
# the marginal distribution of each x_{t} at step t in the diffusion process is
# known. It is x_{t} ~ N(mu_{0}, simga^{2}_t) where sigma_{t} = sigma^{2}_{0} + 
# t * sigma^{2}_q. 
# Here sigma_{q} is the standard deviation of the noise added in the forward
# diffusion process. And times steps t = 0, \delta t, 2 \delta t, ..., T * \delta t
# where \delta t = 1/ T (T being the total number of time steps). 
# In this the score function \delta_{x} p_{t}(x) is known. So in this script, 
# I want to do a sanity check that given, mu_0, sigma^{2}_{0}, sigma^{2}_q and T,
# if the DDPM reverse sampler is used with the known score function to obtain 
# samples of x_{0}, then the histogram of these samples should match the true 
# distribution N(mu_{0}, sigma^{2}_{0}). This should be true for any choice of
# sigma^{2}_q and T too. 
# Also added DDIM sampler which takes a deterministic step in the reverse sampler
# The deterministic mapping is x_{t - \deltat} = x_{t} * 
# \lambda (E[x_{t-\delta_t}| x_{t}] - x_{t}). The expectation is computed directly.
# The score function can also be used to get the expectation

import numpy as np
import matplotlib.pyplot as plt
import os

def score_function(x_t, t, mu_0, sigma2_0, sigma2_q):
    """
    Computes the score function of a Gaussian distribution
    N (mu_0, sigma2_t) where sigma2_t = sigma2_0 + t * sigma2_q
    at a given x_t and time t 
    """

    score = - (x_t - mu_0) / (sigma2_0 + t * sigma2_q)

    return score


def reverse_sampler_ddpm(mu_0, sigma2_0, sigma2_q, T):
    """
    It generates a sample x0 by going through T steps in 
    the reverse order as in DDPM and using the above 
    score function. mu_0 and sigma2_0 are given as input
    arguments because they are needed by the score function
    """
    delta_t = 1 / T
    time_steps = np.linspace(1, delta_t, T)
    x = np.random.normal(0, scale=np.sqrt(sigma2_q))
    for t in time_steps:
        noise = np.random.normal(0, scale=np.sqrt(sigma2_q * delta_t))
        x = x + sigma2_q * delta_t * \
            score_function(x, t, mu_0, sigma2_0, sigma2_q) + noise

    return x


def expected_xt_deltat_given_xt(xt, mu_0, sigma2_0, sigma2_q, t, delta_t):
    """
    Computes the E[x_{t-\deltat}| x_{t}] for this toy problem
    when p(x0) = N(mu_0, sigma2_0)
    """
    sigma2_t = sigma2_q * t
    sigma2_t_deltat = sigma2_q * (t - delta_t)

    expectation = ((sigma2_0 + sigma2_t_deltat) * xt + \
                    (sigma2_q * delta_t * mu_0)) / (sigma2_0 + sigma2_t)

    return expectation


def reverse_sampler_ddim(mu_0, sigma2_0, sigma2_q, T):
    """
    It generates a sample x0 by going through T steps in 
    the reverse order as in DDIM. 
    Update: x_{t-\deltat} = x_{t} + \lambda_{t} * ()
    E[x_{t-\deltat}|x_{t}] 
    = x_{t} + \sigm2_q * \delta_{t} * \grad_{x_{t}} \log p_{t}(x_{t}) 
     mu_0 and sigma2_0 are given as input
    arguments because they are needed by the score function
    """
    delta_t = 1 / T
    time_steps = np.linspace(1, delta_t, T)
    x = np.random.normal(0, scale=np.sqrt(sigma2_q))
    sigma_q = np.sqrt(sigma2_q)
    lambda_step = lambda t: sigma_q * np.sqrt(t) / (sigma_q * np.sqrt(t - delta_t) + sigma_q * np.sqrt(t))
    for t in time_steps:
        # x = x + lambda * (E[x_{t-\deltat}| x_{t}] - x_{t})

        # by explicity calculating the expectation
        x = x + lambda_step(t) * (expected_xt_deltat_given_xt(x, mu_0, sigma2_0, sigma2_q, t, delta_t) - x)

        # by using the score function to get the expectation
        #x = x + lambda_step(t) * sigma2_q * delta_t * \
        #    score_function(x, t, mu_0, sigma2_0, sigma2_q)

    return x


if __name__ == '__main__':

    # parameters of original normal distribution of x0
    mu_0, sigma2_0 = 3, 0.5

    # parameters of the reverse sampling process
    sigma2_q = 50
    T = 1000

    # nos. of samples to obtain from reverse sampling
    nsamples = 1000
    x0_samples_ddpm = []
    x0_samples_ddim = []

    for i in range(nsamples):
        
        # use DDPM
        x0 = reverse_sampler_ddpm(mu_0, sigma2_0, sigma2_q, T)
        x0_samples_ddpm.append(x0)

        # use DDIM
        x0 = reverse_sampler_ddim(mu_0, sigma2_0, sigma2_q, T)
        x0_samples_ddim.append(x0)
    
    
    # ========
    # plotting
    # ========
    # Evaluate true distribution of x0
    x0_test = np.linspace(mu_0 - 3 * np.sqrt(sigma2_0), mu_0 + 3 * np.sqrt(sigma2_0), 100)
    px0 = 1/np.sqrt(2 * np.pi * sigma2_0) * np.exp(-(x0_test - mu_0)**2/(2 * sigma2_0))

    # evaluate noise distribution that you start from
    xbase_test = np.linspace(- 3 * np.sqrt(sigma2_q), 3 * np.sqrt(sigma2_q), 100)
    pxbase = 1/np.sqrt(2 * np.pi * sigma2_q) * np.exp(-(xbase_test)**2/(2 * sigma2_q))

    plt.figure(figsize=(10, 7))
    plt.subplot(1, 3, 1)
    plt.plot(xbase_test, pxbase, label=r'$p(x_{T})$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(x)$')
    plt.title(r'Base distrbution')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(x0_samples_ddpm, density=True, label=r'Samples')
    plt.plot(x0_test, px0, label=r'Data dist. $p(x_{0})$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(x)$')
    plt.title(r'DDPM')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(x0_samples_ddim, density=True, label=r'Samples')
    plt.plot(x0_test, px0, label=r'Data dist. $p_{0}(x_{0})$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(x)$')
    plt.title(r'DDIM')
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join('figures', 'experiment-0-samples-ddm-ddim-toy-example.jpg'))
    
