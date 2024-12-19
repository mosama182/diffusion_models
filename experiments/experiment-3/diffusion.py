import torch
import numpy as np

class Schedule:
    def __init__(self, sigma2_q, T) -> None:
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
    sigma2_q : scaler variance of base distribution
    """
    t = schedule.sample_batch(x0) # t shape: (batch_size, )

    eta_t = torch.sqrt(schedule.sigma2_q * t).unsqueeze(1) * torch.randn_like(x0)
    xt = x0 + eta_t # x_{t}
    
    epsilon = np.sqrt(schedule.sigma2_q * schedule.delta_t) * torch.rand_like(xt)
    xt_deltat = xt + epsilon # x_{t + \delta}

    return xt, xt_deltat, t + schedule.delta_t # t + \deltat will be in interval [0, 1]


if __name__ == "__main__":

    ##########################################
    # test generate_training_sample function #
    ##########################################
    x0 = torch.randn((10, 2))
    sigma2_q = 10
    T = 1000
    schedule = Schedule(sigma2_q, T)
    xt, xt_deltat, t_deltat = generate_training_sample(x0, schedule)
    print(xt.shape, xt_deltat.shape, t_deltat.shape)