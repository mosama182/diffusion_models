"""
Sample using learned flow model
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import FlowField
from data import SyntheticData


def sample(model, T, source_samples):
    """
    source_samples: N x input_dim
    """
    delta_t = torch.tensor(1 / T, dtype=torch.float)
    t_axis = torch.linspace(1, delta_t, T) 

    if isinstance(source_samples, np.ndarray):
        x1 = torch.tensor(source_samples, dtype=torch.float)
    else:
        x1 = source_samples
        
    trajectory = []
    with torch.no_grad():
        for t in t_axis:
            trajectory.append(x1.numpy())
            t_full = t * torch.ones((x1.shape[0], 1), dtype=torch.float)
            x1 = x1 + model(x1, t_full) * delta_t 
            
    x1 = x1.numpy()
    trajectory.append(x1)

    return x1, trajectory, t_axis.numpy()


def visualize_movement(trajectory, t_axis):
    """
    trajectory: a list of numpy array from time t in t_axis
    """
    idx = 10
    t = t_axis[idx]
    x1 = trajectory[idx]
    num_samples = x1.shape[0]
    #color = np.random.randint(256, size=(num_samples, 3))
    #color = color / 255

    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1])
    plt.grid()
    plt.title(f'Points at time $t={t}$')
    plt.show()


if __name__ == "__main__":

    # load model ckpt
    root = os.path.dirname(__file__)
    model = FlowField()
    ckpt = torch.load(os.path.join(root, 'models', 'model.pth'), weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model = model.to("cpu")


    # source data
    num_samples = 500
    dataset = SyntheticData(num_samples, inner_radius=2.5, outer_radius=3.0)

    # sample using flow matching
    T = 100
    target_data, trajectory, t_axis = sample(model, T, dataset.source_data)

    visualize_movement(trajectory, t_axis)

    """
    plt.figure()
    plt.scatter(dataset.source_data[:, 0], dataset.source_data[:, 1], label=r'source samples')
    plt.scatter(target_data[:, 0], target_data[:, 1], label=r'Samples from flow')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid()
    plt.legend()
    plt.show()
    """
    
