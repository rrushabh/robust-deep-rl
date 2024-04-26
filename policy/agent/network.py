import torch
from torch import nn

import utils

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_shape),
									nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)

        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


