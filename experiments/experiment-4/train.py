"""
Training script for learning marginal flow v_{t}
when using pointwise linear flow from transforming
annular distribution to a spiral distribution
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data import SyntheticData
from model import FlowField
from flow import LinearFlow


def train(dataloader, flow, model, optimizer, epochs, device, model_dir):
    """
    source_data: N x input_dim tensor or numpy array
    target_data: N x input_dim tensor or numpy array
    """
    
    model = model.to(device)
    model.train()

    losses = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        for x_source, x_target in dataloader:
            # set gradient to zero
            optimizer.zero_grad()

            # sample time uniformly in [0, 1]
            #t = torch.rand(x_source.shape[0], 1).to(device)
            t = (torch.rand(1, device=x_source.device) + torch.arange(len(x_source), device=x_source.device) / len(x_source)) % (1 - 1e-5)
            t = t.reshape(-1, 1)
            
            # run flow to obtain xt
            xt = flow.runflow(x_target, x_source, t)

            # predict pointwise flow for each (xo, x1) pair at time 't'
            vt_hat = model(xt, t)
            vt = flow.compute(x_target, x_source, t)

            # loss and backpropagation
            loss = torch.mean((vt - vt_hat)**2)
            loss.backward()
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()
            
        epoch_loss /= len(dataloader)
        losses += [epoch_loss]

        # print loss
        #if (epoch + 1) % 50 == 0:
        #    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")


    # save model ckpt
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
        }
    
    os.makedirs(model_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(model_dir, 'model_2.pth'))

    return losses


if __name__ == "__main__":
    
    # read configuration file
    root = os.path.dirname(__file__)
    yaml_file = os.path.join(root, "confg.yaml")
    with open(yaml_file, 'r') as file:
        confg_data = yaml.safe_load(file)
        
    # device
    device = "mps" if torch.cuda.is_available() else "cpu"
    
    # data
    num_samples = confg_data['ntrain']
    dataset = SyntheticData(num_samples, inner_radius=confg_data['inner_rad'], 
                            outer_radius=confg_data['outer_rad'])
    dataloader = DataLoader(dataset, batch_size=confg_data['batch_size'])

    # model
    model = FlowField()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(confg_data['lr']))

    # flow
    flow = LinearFlow()

    # train
    num_epochs = confg_data['epochs']
    model_dir = "models"
    losses = train(dataloader, flow, model, optimizer, num_epochs, 
                    device, model_dir)
    
    # plot training loss
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel(r'epochs')
    plt.ylabel(r'MSE loss training.')
    plt.grid()
    #plt.show()

    # save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/train_loss.jpg')
    

    