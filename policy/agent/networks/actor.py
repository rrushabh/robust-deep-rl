import torch
from torch import nn

import utils

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True))

        self.direction = nn.Sequential(nn.Linear(hidden_dim, 1),
									   nn.Tanh())
        self.gas = nn.Sequential(nn.Linear(hidden_dim, 1),
								 nn.Sigmoid())
        self.brake = nn.Sequential(nn.Linear(hidden_dim, 1),
								   nn.Sigmoid())

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        repr = self.policy(obs)
        mu = torch.stack((self.direction(repr), self.gas(repr), self.brake(repr)), dim=0)

        std = torch.ones_like(mu) * std
        # TODO: does calling truncated normal on a 3-part action work?
        dist = utils.TruncatedNormal(mu, std)
        return dist
