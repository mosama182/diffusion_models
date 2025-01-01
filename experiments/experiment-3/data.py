import os

import torch
import numpy as np
import yaml

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

class SwissRoll(Dataset):

    def __init__(self, tmin, tmax, N, center=(0, 0), scale=1.0) -> None:
        
        t = torch.linspace(tmin, tmax, N)

        center = torch.tensor(center).unsqueeze(0) # shape after unsqueeze (1, 2)

        # diving by tmax to ensure that the radius is 1 and can be controlled by scale variable
        self.vals = center + scale * torch.stack([t * torch.cos(t) / tmax, t * torch.sin(t) / tmax]).T
        #self.vals = torch.tensor([[1.0, 1.0], [-1.0, -1.0], 
        #                          [1.0, -1.0], [-1.0, 1.0], 
        #                          [0.0, 0.0]], dtype=torch.float).reshape(-1, 2)

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]
    

if __name__ == "__main__":

    ###################
    # test data class #
    ###################

    # read configuration file
    root = os.path.dirname(__file__)
    yaml_file = os.path.join(root, "confg.yaml")
    with open(yaml_file, 'r') as file:
        confg_data = yaml.safe_load(file)

    ndata = confg_data['ndata']
    scale = confg_data['scale']
    dataset = SwissRoll(np.pi/2, 4 * np.pi, ndata, scale=scale)


    plt.figure()    
    plt.scatter(dataset.vals[:, 0], dataset.vals[:, 1])
    plt.grid()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Swiss roll synthetic data')
    plt.show()

    # save data figure
    #root = os.path.dirname(__file__)
    #fig_dir = os.path.join(root, 'figures')
    #os.makedirs(fig_dir, exist_ok=True)
    #plt.savefig(os.path.join(fig_dir, 'data.jpg'))

    
