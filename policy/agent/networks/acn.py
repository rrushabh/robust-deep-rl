import torch
from torch import nn

import utils

class ACN(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.acn = nn.Sequential(nn.Linear(repr_dim + action_shape, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, 1),
                                     nn.Sigmoid()
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # TODO: Define the forward pass 
        x = torch.cat([obs, action], dim=1)
        
        confidence = self.acn(x)
        
        return confidence