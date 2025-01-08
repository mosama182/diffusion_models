import torch
from torch import nn


class FlowField(nn.Module):
    """
    Input: xt: RunFlow(v, x1, t) obtained from running flow v on x1 until time t
               (N x input_dim) tensor
           t is time (N, 1) tensor
    
    Output: Predicted flow v at time t shape (N x input_dim) (flow at N different time instants) 
    """

    def __init__(self, input_dim=2, num_hidden=512, num_freqs=10) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_freqs = num_freqs

         # input_dim + 1 because of time embedding (which will just be 't' itself)
        self.net = nn.Sequential(nn.Linear(input_dim + 2 * num_freqs, num_hidden), 
                                 nn.LeakyReLU(), 
                                 nn.Linear(num_hidden, num_hidden), 
                                 nn.LeakyReLU(), 
                                 nn.Linear(num_hidden, num_hidden), 
                                 nn.LeakyReLU(), 
                                 nn.Linear(num_hidden, num_hidden),
                                 nn.LeakyReLU(), 
                                 nn.Linear(num_hidden, num_hidden), 
                                 nn.LeakyReLU(), 
                                 nn.Linear(num_hidden, input_dim), 
                                 nn.LeakyReLU(), 
                                 nn.Linear(input_dim, input_dim))
        
    
    def time_embedding(self, t):
        freq = 2 * torch.arange(self.num_freqs, device=t.device) * torch.pi
        t = freq * t 
        return torch.cat((t.cos(), t.sin()), dim=-1)

        
    def forward(self, xt, t):
        """
        xt: (N, input_dim) tensor
        t : (N, 1) tensor
        """
        time_embed = self.time_embedding(t)
        x_input = torch.cat((xt, time_embed), dim=1)
        
        return self.net(x_input)
    

if __name__ == "__main__":
    xt = torch.tensor([1, 1], dtype=torch.float).reshape(1, 2)
    t = torch.rand(1, 1)

    model = FlowField()
    print(model(xt, t))
