"""
Sample using learned flow model
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from celluloid import Camera

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


def visualize_movement(trajectory):
    """
    trajectory: a list of numpy array from time t in t_axis
    """
    x1 = trajectory[0]
    colors = np.arctan(x1[:, 1] / x1[:, 0])
    fig, ax = plt.subplots(figsize=(4, 4))
    camera = Camera(fig)
    ax.set_title("Toy example - flow matching")

    for idx in range(len(trajectory)):
        x = trajectory[idx]
        plt.scatter(x[:, 0], x[:, 1], c=colors)
        camera.snap()
    
    anim = camera.animate(blit=True)

    os.makedirs('figures', exist_ok=True)
    anim.save('figures/scatter.gif', writer='ffmpeg', fps=7)

    
if __name__ == "__main__":

    # load model ckpt
    root = os.path.dirname(__file__)
    model = FlowField()
    ckpt = torch.load(os.path.join(root, 'models', 'model.pth'), weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model = model.to("cpu")

    # read configuration file
    root = os.path.dirname(__file__)
    yaml_file = os.path.join(root, "confg.yaml")
    with open(yaml_file, 'r') as file:
        confg_data = yaml.safe_load(file)
        
    # source data
    num_samples = confg_data['ntest']
    dataset = SyntheticData(num_samples, inner_radius=confg_data['inner_rad'], 
                            outer_radius=confg_data['outer_rad'])

    # sample using flow matching
    T = confg_data['T']
    target_data, trajectory, t_axis = sample(model, T, dataset.source_data)

    visualize_movement(trajectory)

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
    
