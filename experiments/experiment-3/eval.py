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

if __name__ == "__main__":

    
    # read configuration file
    root = os.path.dirname(__file__)
    yaml_file = os.path.join(root, "confg.yaml")
    with open(yaml_file, 'r') as file:
        confg_data = yaml.safe_load(file)

    # underlying spiral/Swiss roll data
    ndata = 1000
    dataset = SwissRoll(np.pi/2, 5 * np.pi, ndata)

    # load model ckpt
    root = os.path.dirname(__file__)
    model = TimeInputMLP()
    ckpt = torch.load(os.path.join(root, 'models', 'model.pth'), weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # sampler
    sampler = DDIMSampler(sigma2_q=confg_data['sigma2_q'], T=confg_data['T'])

    nsamples = 1
    samples = []
    trajectories = []
    print(f"Drawing samples from DDIM sampler")
    for _ in tqdm(range(nsamples)):
        x0, trajectory = sampler.sample(model)
        samples.append(x0)
        trajectories.append(trajectory)
    
    samples = np.array(samples).reshape(-1, 2)
    print(samples)

    # plot 
    #plt.figure()
    plt.scatter(dataset.vals[:, 0], dataset.vals[:, 1], label=r'Spiral')
    plt.scatter(samples[:, 0], samples[:, 1], label=r'Samples DDIM')
    plt.grid()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    #plt.legend()
    #plt.xlim([-3, 3])
    #plt.ylim([-3, 3])
    plt.show()
