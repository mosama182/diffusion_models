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
The model will be trained to estimate E[x_{0} | x_{t}]. 
We know that E[Y|X] is the MMSE estimator. Here Y = x_{0} and 
X = [x_{t}, t].  
So the input to the model will be x_{t}, t
and the output will \hat{x_0} and we will be minimizing the mean
square error. 

**Note:** Originally, I wanted to estimate E[x_{t} | x_{t + \deltat}]
          but was not successful in doing that. So I estimate
          E[x_{0} | x_{t}] and use the relationship between  
          E[x_{t} | x_{t + \deltat}] and E[x_{0} | x_{t}] to obtain
          former from the later. 
          **Todo** I will investigate later why I was failing to 
          estimate the original quantity directly        

#########################
GETTING A TRAINING SAMPLE
#########################
Given a datapoint x_{0}, we need to produce a training sample 
(x_{t}, t) and for this we will be
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
        
        #epsilon ~ N (0, \sigma2_q * \deltat) -- ignored 
        #x_{t + \deltat} = x_{t} + epsilon  -- ignored

        
########
NN MODEL
########
The model is simply a fully connected NN. It has 3 layers with 128 neurons per
layer and ReLU activation. The raw input to the model x_{t} and t
are converted to a 3-dimensional vector. The first 2 dimensions are simply the raw 
x_{t} itself and the next 1 dimensions is a transformed times embedding
\phi(t) = [t]. It can be something else as well e.g. 
\phi(t) = [sin(2 * pi * (t)), cos(2 * pi * (t))]. 
So the first fully connected layer has input dimension of 3 (in the later case 4). 
The output layer has dimenson 2, and estimated \hat{x_{0}}

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
                  model_dir  : str='',
                  device     : str='mps'):
    
    # training
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_epochs = []
    for epoch in tqdm(range(epochs)):
        batch_loss = 0
        for x0 in dataloader:
            
            optimizer.zero_grad()
            xt, t = generate_training_sample(x0, schedule)

            # move to gpu
            x0 = x0.to(device)
            xt = xt.to(device)
            t = t.to(device)
            
            # backpropagation
            x0_hat = model(xt, t)
            loss = nn.MSELoss()(x0_hat, x0)
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
    scale = confg_data['scale']
    dataset = SwissRoll(np.pi/2, 4 * np.pi, ndata, scale=scale)
    batch_size = confg_data['batch_size']
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

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
    plt.title(r'Learning $E[x_o | x_t]$ from data')
    #plt.show()

    # save train loss
    fig_dir = os.path.join(root, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'train_loss.jpg'))

    