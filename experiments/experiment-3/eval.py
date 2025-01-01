"""
This script will load the trained model ckpt 
for the Swiss roll dataset. It will then draw 
samples from the DDIMSampler. What we expect to see is 
that samples should lie close to the underlying spiral
from which the training data was generated. 
"""
import os

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from data import SwissRoll
from model import TimeInputMLP
from sampler import DDIMSampler
from tqdm import tqdm

from diffusion import generate_training_sample, Schedule

if __name__ == "__main__":

    
    # read configuration file
    root = os.path.dirname(__file__)
    yaml_file = os.path.join(root, "confg.yaml")
    with open(yaml_file, 'r') as file:
        confg_data = yaml.safe_load(file)

    # underlying spiral/Swiss roll data
    ndata = confg_data['ndata']
    scale = confg_data['scale']
    dataset = SwissRoll(np.pi/2, 4 * np.pi, ndata, scale=scale)

    # load model ckpt
    root = os.path.dirname(__file__)
    model = TimeInputMLP()
    ckpt = torch.load(os.path.join(root, 'models', 'model.pth'), weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    #model = model.to("cpu")

    
    # sampler
    sampler = DDIMSampler(sigma2_q=confg_data['sigma2_q'], T=confg_data['T'])

    nsamples = 500
    samples = []
    trajectories = []
    print(f"Drawing samples from DDPM sampler")
    for _ in tqdm(range(nsamples)):
        x0, trajectory = sampler.sample(model)
        samples.append(x0)
        trajectories.append(trajectory)
    
    samples = np.array(samples).reshape(-1, 2)

    # plot 
    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], label=r'Samples DDPM')
    plt.scatter(dataset.vals[:, 0], dataset.vals[:, 1], label=r'Original data points')
    plt.grid()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.title(f'Samples when learning $E[x_0 | x_t]$ from data')

    #plt.show()

    # save figure
    fig_dir = os.path.join(root, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'eval_ddpm_samples.jpg'))
    

    """
    x0 = torch.tensor([1.0, 1.0], dtype=torch.float).reshape(-1, 2)
    schedule = Schedule(confg_data['sigma2_q'], confg_data['T'])
    xt_list = []
    xt_pred_list = []
    error_list = []
    for _ in range(1000):
        xt, xt_deltat, t_deltat = generate_training_sample(x0, schedule)
        xt_list.append(xt.detach().numpy())

        xt_pred = model(xt_deltat, t_deltat)
        xt_pred_list.append(xt_pred.detach().numpy())

        error = torch.sum((xt_pred - xt)).item()

        error_list.append(error)

    xt_list = np.array(xt_list).reshape(-1, 2)
    xt_pred_list = np.array(xt_pred_list).reshape(-1, 2)

    plt.figure()
    plt.scatter(xt_list[:, 0], xt_list[:, 1], label=r'truth')
    plt.scatter(xt_pred_list[:, 0], xt_pred_list[:, 1], label=r'pred')
    plt.grid()
    plt.show()

    plt.figure()
    plt.hist(error_list)
    plt.show()
    
    
    
    schedule = Schedule(confg_data['sigma2_q'], confg_data['T'])
    x1 = torch.tensor([2.56, -0.14], dtype=torch.float).reshape(-1, 2)
    x1 = np.sqrt(schedule.sigma2_q) * torch.rand_like(x1)
    t_axis = torch.linspace(1, schedule.delta_t, schedule.T)
    print(f"x1: {x1}")

    for t in t_axis:
        t = torch.tensor([t])
        x = model(x1, t)
        print(x)
        x1 = x + np.sqrt(schedule.sigma2_q * schedule.delta_t) * torch.rand_like(x)

    print(f"x0: {x1}")
    """    


