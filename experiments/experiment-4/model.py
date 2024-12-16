import torch
from torch import nn

class TimeInputMLP(nn.Module):
    def __init__(self, num_input=2, num_hidden=128) -> None:
        super().__init__()
        
        # Three layers with ReLU activation
        # Adding 2 to num_input because time 
        # will be mapped to a 2-dimensional embedding
        layers = []
        layers.extend([nn.Linear(num_input + 2, num_hidden), nn.ReLU()])
        layers.extend([nn.Linear(num_hidden, num_hidden), nn.ReLU()])
        layers.extend([nn.Linear(num_hidden, num_hidden)], nn.ReLU())
        num_output = num_input
        layers.append(nn.Linear(num_hidden, num_output))

        self.net = nn.Sequential(*layers)

    def time_embed(self, t):
        # t has shape (batch_size, )
        t = t.unsqueeze(1)
        time_embed = torch.cat([torch.cos(2 * torch.pi * t), torch.sin(2 * torch.pi * t)], dim=1)
        
        return time_embed
    
    def forward(self, x, t):
        """
        x : (batch_size, 2) # x_{t + \deltat}
        t: (batch_size, )   # t + \deltat
        """
        time_embed = self.time_embed(t) # (batch_size, 2)
        model_input = torch.cat([x, time_embed], dim=1) # (batch_size, 4)

        return self.net(model_input) # \hat{x_{t}}
    
