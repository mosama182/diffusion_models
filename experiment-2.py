# This experiment plots trajectories of the samples
# obtained from DDIM but considers the data distribution
# which has probability weights at two data points
# Problem setup:
# data distribution: p_{0}(x_{0}) = 0.5 * \delta(x0 - a) + 0.5 * \delta(x0 - b), x0 \in R^{2}
# E[x_[t-\deltat] | xt] = p(x0 = a | xt) * E[x_[t-\deltat] | xt, x0 = a] 
#                       + p(x0 = b | xt) * E[x_[t-\deltat] | xt, x0 = b].
# Here,
# E[x_[t-\deltat] | xt, x0 = a] = \sigma2_{t-\deltat} / \sigma2_t  * xt + (\sigma2_t - \sigma2_{t-\deltat}) / \sigma2_t * a
# p(x0 = a | xt) = p0(x0 = a) p(xt | x0 = a) / p_{t}(xt),
# where p(xt | x0 = a) = N(xt; a, \sigma2_t), p_t(xt) = 0.5 *  N(xt; a, \sigma2_t) + 0.5 * N(xt; b, \sigma2_t).
# DDIM reverse sampling:
# x_{t - \deltat} = xt +  \lambda * (E[xt - \deltat | xt] - xt)
# Expression for 
# \lambda * (E[xt - \deltat | xt] - xt) = p(x0 = a | xt) * \lambda * (E[x_[t-\deltat] | xt, x0 = a] - xt)
#                                       + p(x0 = b | xt) * \lambda * (E[x_[t-\deltat] | xt, x0 = b] - xt),
# Here
# \lambda * (E[x_[t-\deltat] | xt, x0 = a] - xt) = (1 - sigma_{t - \deltat} / sigma_{t}) * (a - xt)

import os

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random


def single_update_ddim(x0=np.zeros((1, 2)), xt=None, t=None, sigma2_q=None, delta_t=None):
    """
    Computes 
    \lambda * (E[xt - \deltat | xt, x0 = a] - xt) 

    xt : 1 x 2 numpy array
    x0 : 1 x 2 numpy array 
    """
    sigma_t = np.sqrt(sigma2_q * t)
    sigma_t_deltat = np.sqrt(sigma2_q * (t - delta_t))

    update = (1 - sigma_t_deltat / sigma_t) * (x0 - xt)

    return update


def pt_xt_given_x0(xt=np.zeros((1, 2)), x0=np.zeros((1, 2)), sigma2_q=None, t=None):
    """
    Compute pt(xt | x0 = a) = N(xt; a, sigma2_t)
    xt: 1 x 2 numpy array
    x0 : 1 x 2 numpy array
    """

    sigma2_t = sigma2_q * t
    cov = sigma2_t * np.eye(2)
    mean = x0
    mv_gaussian = multivariate_normal(mean=mean, cov=cov)

    return mv_gaussian.pdf(xt)


def prob_x0_given_xt(x0_values=[], p0_values=[], x0=np.zeros((1, 2)), p0=0,  
                     xt=np.zeros((1, 2)), sigma2_q=None, t=None):
    """
    Compute p(x0 = a | xt) = p0(x0 = a) p(xt | x0 = a) / p_{t}(xt)
    x0_values: list of all possible values of x0
    p0_values: list of all probability values of p0 for different x0
    x0_index: index of the specific value of x0 for which the posterior probabilyt
              is to be calculated
    """
    prob_xt_given_x0 = pt_xt_given_x0(xt=xt, x0=x0, sigma2_q=sigma2_q, t=t)
    prob_xt = 0
    for idx, x0 in enumerate(x0_values):
        prob_xt += p0_values[idx] * pt_xt_given_x0(xt=xt, x0=x0, sigma2_q=sigma2_q, t=t)
    
    post_prob_x0 = p0 * prob_xt_given_x0 / prob_xt

    return post_prob_x0


def combined_update_ddim(x0_values=[], p0_values=[], xt=np.zeros((1, 2)), 
                         t=None, sigma2_q=None, delta_t=None):
    """
    Computed the total update of ddim reverse sampling using 
    the individual update and probabilty information
    """
    total_update = np.zeros((1, 2))
    for x0, p0 in zip(x0_values, p0_values):
        post_prob_x0 = prob_x0_given_xt(x0_values=x0_values, p0_values=p0_values, x0=x0, p0=p0, 
                                        xt=xt, sigma2_q=sigma2_q, t=t)
        
        single_update = single_update_ddim(x0=x0, xt=xt, t=t, sigma2_q=sigma2_q, delta_t=delta_t)

        total_update += post_prob_x0 * single_update
    
    return total_update


def reverse_sample_ddim(x0_values=[], p0_values=[], sigma2_q=None, T=None):
    """
    Performs DDIM reverse sampling
    """
    trajectory = [] # list to save trajectory
    delta_t = 1 / T
    time_steps = np.linspace(1, delta_t, T)
    x = np.random.multivariate_normal(mean=[0, 0], cov=[[sigma2_q, 0], [0, sigma2_q]], size=1) # x1
    trajectory.append(x.tolist())
    for t in time_steps:
        update = combined_update_ddim(x0_values=x0_values, p0_values=p0_values, 
                                      xt=x, t=t, sigma2_q=sigma2_q, delta_t=delta_t)
        x += update
        trajectory.append(x.tolist())
        
    trajectory = np.array(trajectory).reshape(-1, 2)

    return x, trajectory


if __name__ == "__main__":

    # data distribution
    x0_values = [np.array([2, 2]), np.array([-2, 2])]
    p0_values = [0.5, 0.5]
    
    # forward diffusion process parameters
    sigma2_q = 100
    T = 50
    
    # =====================
    # reverse DDIM sampling
    # =====================
    # collect trajectories
    trajectories = []
    n_traj = 10

    for i in range(n_traj):
        _, trajectory = reverse_sample_ddim(x0_values=x0_values, p0_values=p0_values, sigma2_q=sigma2_q, T=T)
        trajectories.append(trajectory)
    
    # =================
    # plot trajectories
    # =================
    plt.figure()

    for trajectory in trajectories:
        random_color = (random.random(), random.random(), random.random())  # Generate random RGB color
        plt.scatter(trajectory[:, 0], trajectory[:, 1], color=random_color, marker='x')
    
    # plot data distribution
    plt.scatter(x0_values[0][0], x0_values[0][1], color='red', marker='o')
    plt.scatter(x0_values[1][0], x0_values[1][1], color='red', marker='o', label=r'$p_{0}(x_{0}) = 0.5 * \delta(x_{0} - [2, 2]) + 0.5 * \delta(x_{0} - [-2, 2])$')

    plt.grid()
    plt.xlabel(r'$x_{1}$')
    plt.ylabel(r'$x_{2}$')
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.title('Trajectoreis from DDIM with toy data distribution $p_{0}(x_{0})$')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join('figures', 'experiment-2-sample-trajectories-ddim.jpg'))    
    #plt.show()