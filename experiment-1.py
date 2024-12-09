# This experiment plots trajecotories of points from x1, x1-\deltat, ..., x0
# when using DDIM sampling.
# Problem setup:
# p0(x0) = \delta(x0 - a) (a dirac delta at 'a') where x0 \in R^{2}
# E[xt - \deltat | xt] = 
# (sigma2_{t - \deltat} * xt + sigma2_q * \deltat * a) / sigma2_{t}
# \lambda * (E[xt - \deltat | xt] - xt) = 
# (1 - sigma_{t - \deltat} / sigma_{t}) * (a - xt)
# \lamda = \sqrt(t) / (\sqrt(t - \deltat) + \sqrt(t))
# DDIM reverse sampling:
# x_{t - \deltat} = xt +  \lambda * (E[xt - \deltat | xt] - xt)   

import numpy as np
import matplotlib.pyplot as plt
import random

def update_ddim(a, xt, t, sigma2_q, delta_t):
    """
    Computes 
    \lambda * (E[xt - \deltat | xt] - xt) 

    xt : 2 x 1 numpy array
    a : 2 x 1 numpy array 
    """
    sigma_t = np.sqrt(sigma2_q * t)
    sigma_t_deltat = np.sqrt(sigma2_q * (t - delta_t))

    update = (1 - sigma_t_deltat / sigma_t) * (a - xt)

    return update


def reverse_sample_ddim(a, sigma2_q, T):
    """
    Performs DDIM reverse sampling
    """
    trajectory = [] # list to save trajectory
    delta_t = 1 / T
    time_steps = np.linspace(1, delta_t, T)
    x = np.random.multivariate_normal(mean=[0, 0], cov=[[sigma2_q, 0], [0, sigma2_q]], size=1) # x1
    x = x.reshape(2, 1)
    trajectory.append(x.tolist())
    for t in time_steps:
        update = update_ddim(a, x, t, sigma2_q, delta_t)
        x += update
        trajectory.append(x.tolist())
    
    trajectory = np.array(trajectory).reshape(-1, 2)

    return x, trajectory


if __name__ == "__main__":

    a = np.array([2, 3]).reshape(-1, 1)
    sigma2_q = 20
    T = 20

    # ====================
    # collect trajectories
    # ====================
    trajectories = []
    n_traj = 10

    for i in range(n_traj):
        _, trajectory = reverse_sample_ddim(a, sigma2_q, T)
        trajectories.append(trajectory)

    # =================
    # plot trajectories
    # =================
    plt.figure()
    for trajectory in trajectories:
        random_color = (random.random(), random.random(), random.random())  # Generate random RGB color
        plt.scatter(trajectory[:, 0], trajectory[:, 1], color=random_color, marker='x')
    
    # plot data distribution
    plt.scatter(a[0], a[1], color='red', marker='o', label=r'$p_{0}(x_{0})=\delta(x_{0} - [2, 2])$')
    
    plt.grid()
    plt.xlabel(r'$x_{1}$')
    plt.ylabel(r'$x_{2}$')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.legend()
    plt.title('Trajectoreis from DDIM with toy data distribution $p_{0}(x_{0})$')
    plt.savefig(f'experiment-1-sample-trajectories-ddim.jpg')
    
    