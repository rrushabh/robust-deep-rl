import torch
import torch.nn.functional as F

import utils
from agent.networks.actor import Actor

class Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
    ):
        self.device = device
        self.lr = lr
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.bc_weight = 0.1
        
        self.actor = Actor(obs_shape[0], action_shape[0], hidden_dim)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.train()
