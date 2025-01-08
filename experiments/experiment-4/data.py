import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset

class SyntheticData(Dataset):
    """
    Synthetic data class.
    Here the source distribution is annular 
    And the target distribtion is spiral
    """

    def __init__(self, num_samples, inner_radius=1, outer_radius=2, 
                 num_turns=3, noise=0.1) -> None:
        
        self.num_samples = num_samples
        self.source_data = torch.tensor(self.generate_annular(num_samples, inner_radius, outer_radius), dtype=torch.float)
        self.target_data = torch.tensor(self.generate_spiral(num_samples, num_turns, noise), dtype=torch.float)

    # Generate annular distribution: source distribution
    def generate_annular(self, num_samples, inner_radius, outer_radius):
        r = np.random.uniform(inner_radius, outer_radius, num_samples)
        theta = np.random.uniform(0, 2 * np.pi, num_samples)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)

    # Generate spiral distribution: target distribution
    def generate_spiral(self, num_samples, num_turns, noise):
        theta = np.linspace(0, num_turns * 2 * np.pi, num_samples)
        r = theta / (2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        data = np.stack([x, y], axis=1)
        noise = np.random.normal(0, noise, data.shape)
        return data + noise
    
    def __len__(self):
        return len(self.source_data)
    
    
    def __getitem__(self, i):
        return self.source_data[i], self.target_data[i]


if __name__ == "__main__":

    num_samples = 1000
    dataset = SyntheticData(num_samples, inner_radius=2, outer_radius=2.25)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(dataset.source_data[:, 0], dataset.source_data[:, 1], label=r'source distr.')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid()
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(dataset.target_data[:, 0], dataset.target_data[:, 1], label=r'target distr.')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.grid()
    plt.legend()

    plt.show()