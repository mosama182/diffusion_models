"""
#######
DATASET
#######
This experiment is to learn a diffusion model
for a toy Swissroll dataset. And then use the 
learned model in DDIM reverse sampler to see
the trajectories and learned data points. The data
is in R^{2}.

#######################
WHAT WE AIM TO ESTIMATE
#######################
The model will be trained to estimate E[x_t | x_{t + \delat}]. 
We know that E[Y|X] is the MMSE estimator. Here Y = x_{t} and 
X = [x_{t + \deltat}, t + \deltat].  
So the input to the model will be x_{t + \deltat}, t + \deltat 
and the output will \hat{x_t} and we will be minimizing the mean
square error.

#########################
GETTING A TRAINING SAMPLE
#########################
Given a datapoint x_{0}, we need to produce a training sample 
(x_{t}, x_{t + \deltat}, t + \deltat) and for this we will be
creating two functions/classes:
    1. Schedule: is a class that defines N fixed time points
    in interval [0, 1]. When generating a training sample, we call
    the schedule to give us value of 't' which is randomly sampled from 
    the N fixed points 
    
    2. generate_training_sample(x0, sigma2_q, \deltat, Schedule): It calls the schedule to 
    get a value of t and then produces training sample a/c to:
        t = Schedule(x0)
        \eta_{t} ~ N (0, sigma2_q * t * I)
        x_{t}  = x_{0} + \eta_{t}
        epsilon ~ N (0, \sigma2_q * \deltat) 
        x_{t + \deltat} = x_{t} + epsilon  

########
NN MODEL
########
The model is simply a fully connected NN. It has 3 layers with 128 neurons per
layer and ReLU activation. The raw input to the model x_{t+\deltat} and t+\deltat
are converted to a 4-dimensional vector. The first 2 dimensions are simply the raw 
x_{t + \deltat} itself and the next 2 dimensions is a transformed times embedding
\phi(t+\delta) = [sin(2 * pi * (t + \deltat)), cos(2 * pi * (t + \deltat))]. So the 
first fully connected layer has input dimension of 4. The output layer has dimenson 
2, and estimated \hat{x_{t}}

"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from diffusion import Schedule, generate_training_sample, generate_entire_trajectory
from model import TimeInputMLP
from data import SwissRoll

def training_loop(dataloader : DataLoader, 
                  model      : nn.Module, 
                  schedule   : Schedule,
                  lr         : float = 1e-5,
                  epochs     : int = 10000, 
                  model_dir  : str=''):
    
    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_epochs = []
    for epoch in tqdm(range(epochs)):
        batch_loss = 0
        for x0 in dataloader:
            optimizer.zero_grad()
            #xt, xt_deltat, t_deltat = generate_training_sample(x0, schedule)
            xt, xt_deltat, t_deltat = generate_entire_trajectory(x0, schedule)
            xt_hat = model(xt_deltat, t_deltat)
            loss = nn.MSELoss()(xt_hat, x0)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
        
        loss_epoch = batch_loss / len(dataloader)
        loss_epochs.append(loss_epoch)

        #if epoch % 10 == 0:
        #    print(f"Epoch [{epoch}/{epochs}], Loss: {loss_epoch:.4f}")
        
    # save model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_epoch
        }
    os.makedirs(model_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(model_dir, 'model.pth'))

    return loss_epochs


if __name__ == "__main__":

    # read configuration file
    root = os.path.dirname(__file__)
    yaml_file = os.path.join(root, "confg.yaml")
    with open(yaml_file, 'r') as file:
        confg_data = yaml.safe_load(file)
        
    # dataloader
    ndata = confg_data['ndata']
    dataset = SwissRoll(np.pi/2, 5 * np.pi, ndata)
    print(f"Single data point: {dataset.vals}")
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    # model
    model = TimeInputMLP()
    model.train()
   
    # diffusion forward process parameters
    T = confg_data['T']
    sigma2_q = confg_data['sigma2_q']
    schedule = Schedule(sigma2_q, T)

    # train
    epochs = confg_data['epochs']
    lr = float(confg_data['lr'])
    model_dir = os.path.join(root, 'models')
    loss_epochs = training_loop(dataloader, model, schedule, lr=lr, epochs=epochs, model_dir=model_dir)
    
    #plot training loss
    plt.figure()
    plt.plot(range(epochs), loss_epochs)
    plt.grid()
    plt.xlabel(r'epoch')
    plt.ylabel(r'Training loss')
    #plt.show()

    # save train loss
    fig_dir = os.path.join(root, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'train_loss_gen_entire_traj.jpg'))
    