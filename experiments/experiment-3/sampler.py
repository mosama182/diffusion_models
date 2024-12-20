import os 

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

class DDIMSampler:
    """
    Class for sampling using DDIM
    __int__: The constructor of the class will have 
            parameters T and noise base noise variance
            sigma2_q, \deltat = 1 / T

    update: Computes DDIM update: 
            \lambda * (E[x_{t-\deltat}| x_{t}] - x_{t}),
            where 
            \lambda = \sigma_{t} / (\simga_{t - \deltat} + \sigma_{t}),
            where \sigma_{t} = \sqrt(\sigma2_q * t)

            takes x_{t}, t 
            model (estimates \E[x_{t-\deltat} | x_{t}])
            

    sample: Provide a datasample x_{0} by going over for 
            loop over T time steps
            Start with x_{1}, x_{t - \deltat}, ..., x_{0}
            Given x_{t},
            for t = 1, 1 - \deltat, ..., \deltat
                x_{t - \deltat} = x_{t} + update(x_{t}, t)

            Also saves the trajectory in case it is needed   
    """
    def __init__(self, sigma2_q=10, T=100) -> None:
        self.sigma2_q = sigma2_q
        self.T = T
        self.delta_t = 1 / T

    def _update(self, xt, t, model):
        """
        xt   : 1 x 2 numpy array
        t    : scalar
        model: pytorch model loaded from saved 
               checkpoint or some function 
               that takes x_{t}, t as input 
               and provies an estimate of 
               \hat{E}[x_{t - \deltat} | x_{t}]
        """
        # \hat{E}[x_{t - \deltat} | x_{t}]
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                device = next(model.parameters()).device
                xt_tensor = torch.tensor(xt, dtype=torch.float).to(device)
                t_tensor = torch.tensor(t, dtype=torch.float).reshape((1, )).to(device)
                exp_xt_deltat = model(xt_tensor, t_tensor)
                exp_xt_deltat = exp_xt_deltat.numpy() # 1 x 2 numpy array
        else:
            exp_xt_deltat = model(xt, t) # maybe it is just some function

        sigma_t = np.sqrt(self.sigma2_q * t)
        sigma_t_deltat = np.sqrt(self.sigma2_q * (t - self.delta_t))
        scale = sigma_t / (sigma_t + sigma_t_deltat)
        update = scale * (exp_xt_deltat - xt)

        return update
    
    def sample(self, model):
        """
        DDIM reverse sampling
        model: pytorch model loaded from saved 
               checkpoint or some function 
               that takes x_{t}, t as input 
               and provies an estimate of 
               \hat{E}[x_{t - \deltat} | x_{t}]
        """
        trajectory = []
        time_steps = np.linspace(1, self.delta_t, self.T)
        x = np.random.multivariate_normal(mean=[0, 0], cov=[[self.sigma2_q, 0], [0, self.sigma2_q]], 
                                          size=1) # x1
        x = x.reshape(-1, 2) # force shape (1, 2)

        for t in time_steps:
            trajectory.append(x.tolist())
            update = self._update(x, t, model)
            x += update
        
        trajectory.append(x.tolist())
        trajectory = np.array(trajectory).reshape(-1, 2)

        return x, trajectory
    


if __name__ == "__main__":
   
    # read configuration file
    root = os.path.dirname(__file__)
    yaml_file = os.path.join(root, "confg.yaml")
    with open(yaml_file, 'r') as file:
        confg_data = yaml.safe_load(file)

    sampler = DDIMSampler(sigma2_q=confg_data['sigma2_q'], T=confg_data['T'])
    
    # load model ckpt
    from model import TimeInputMLP
    model = TimeInputMLP()
    ckpt = torch.load(os.path.join(root, 'models', 'model.pth'), weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # single sample
    x0, trajectory = sampler.sample(model)
    print(trajectory)

    #plt.figure()
    #plt.scatter(trajectory[:, 0], trajectory[:, 1])
    #plt.show()

