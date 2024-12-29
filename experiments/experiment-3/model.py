import torch
from torch import nn

class TimeInputMLP(nn.Module):
    def __init__(self, num_input=2, num_hidden=128) -> None:
        super().__init__()
        
        # Three layers with ReLU activation
        # Adding 2 to num_input because time 
        # will be mapped to a 1-dimensional embedding
        layers = []
        layers.extend([nn.Linear(num_input + 1, num_hidden), nn.ReLU()])
        layers.extend([nn.Linear(num_hidden, num_hidden), nn.ReLU()])
        layers.extend([nn.Linear(num_hidden, num_hidden), nn.ReLU()])
        num_output = num_input
        layers.append(nn.Linear(num_hidden, num_output))

        self.net = nn.Sequential(*layers)

    def time_embed(self, t):
        # t has shape (batch_size, )
        t = t.unsqueeze(1)
        #time_embed = torch.cat([torch.cos(2 * torch.pi * t), torch.sin(2 * torch.pi * t)], dim=1)
        time_embed = t

        return time_embed
    
    def forward(self, x, t):
        """
        x : tensor (batch_size, 2) # x_{t + \deltat}
        t: tensor (batch_size, )   # t + \deltat
        """

        time_embed = self.time_embed(t) # (batch_size, 2)
        model_input = torch.cat([x, time_embed], dim=1) # (batch_size, 3)

        return self.net(model_input) # \hat{x_{t}}
    

if __name__ == "__main__":

    xt_delta = torch.randn((10, 2))
    t_delta = torch.rand((10, ))
    
    model = TimeInputMLP()
    model_output = model(xt_delta, t_delta)
    print(model_output.shape)
