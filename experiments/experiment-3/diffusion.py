import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class Schedule:
    def __init__(self, sigma2_q, T) -> None:
        self.T = T
        self.delta_t = 1 / T
        self.sigma2_q = sigma2_q

        # t will have values 0, \deltat, 2\deltat, ..., 1 - \deltaT
        # total T values
        self.time = torch.linspace(0, 1 - self.delta_t, T) 
    
    def __len__(self):
        return len(self.time)
    
    def __getitem__(self, i):
        return self.time[i]

    def sample_batch(self, x0:torch.FloatTensor):
        batch_size = x0.shape[0]
        return self[torch.randint(low=0, high=len(self), size=(batch_size, ))]


def generate_training_sample(x0:torch.FloatTensor, schedule:Schedule):
    """
    x0 shape: batch_size x 2
    """
    t = schedule.sample_batch(x0) # t shape: (batch_size, )

    eta_t = torch.sqrt(schedule.sigma2_q * t).unsqueeze(1) * torch.randn_like(x0)
    xt = x0 + eta_t # x_{t}
    
    epsilon = np.sqrt(schedule.sigma2_q * schedule.delta_t) * torch.rand_like(xt)
    xt_deltat = xt + epsilon # x_{t + \delta}

    return xt, xt_deltat, t + schedule.delta_t # t + \deltat will be in interval [0, 1]


def generate_entire_trajectory(x0:torch.FloatTensor, schedule:Schedule):
    """
    Given a datapoint (or batch) x0, it generates the entire trajectory
    x0, x_{\delta_t}, x_{2\delta_t}, ..., x_{1}. Then we use this trajectory 
    to create matrics X_{t + \deltat} (next x) and X_{t} for training purpose.
    These matrices will be of size (batch_size x T) x 2

    Input:
    x0 shape: batch_size x 2
    """
    T = schedule.T
    sigma2_q = schedule.sigma2_q
    delta_t = schedule.delta_t

    # t = [delta_t, 2_deltat, ..., 1]
    t_axis  = torch.linspace(schedule.delta_t, 1, T).reshape(schedule.T, 1)
    
    # x_deltat, x_2deltat, ..., x1
    Xt_deltat = torch.zeros(size=(T, 2))
    
    # forward process diffusion
    x = x0
    for ti in range(T):
        Xt_deltat[ti] = x + np.sqrt(sigma2_q * delta_t) * torch.randn_like(x)
        x = Xt_deltat[ti]

    Xt = torch.cat((x0, Xt_deltat[:-1, :]), dim=0)

    return Xt, Xt_deltat, t_axis


if __name__ == "__main__":

    ##########################################
    # test generate_training_sample function #
    ##########################################
    x0 = torch.tensor([0.0, 0.0], dtype=torch.float)
    x0 = x0.reshape(-1, 2)
    sigma2_q = 1
    T = 200
    schedule = Schedule(sigma2_q, T)
    
    """
    xt_list = []
    for epoch in tqdm(range(100)):
        for _ in range(T):
            xt, xt_deltat, t_deltat = generate_training_sample(x0, schedule)
            xt_list.append(xt.numpy())

    xt = np.array(xt_list).reshape(-1, 2)
    plt.figure()
    plt.scatter(xt[:, 0], xt[:, 1])
    plt.scatter(x0.numpy()[0], x0.numpy()[1])
    plt.grid()
    plt.show()
    """
    Xt, Xt_deltat, t_deltat = generate_entire_trajectory(x0, schedule)

    print(Xt.shape, Xt_deltat.shape, t_deltat.shape)

    plt.figure()
    plt.plot(Xt[:, 0], Xt[:, 1])
    plt.plot(Xt_deltat[:, 0], Xt_deltat[:, 1])
    plt.show()