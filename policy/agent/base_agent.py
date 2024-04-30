import torch
import torch.nn.functional as F

import utils
from agent.networks.actor import Actor
from agent.networks.acn import ACN
from agent.networks.encoder import Encoder

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
		obs_type
	):
		self.device = device
		self.lr = lr
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.use_encoder = True if obs_type=='pixels' else False
		self.bc_weight = 0.1

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]
		
		self.actor = Actor(repr_dim, action_shape[0], hidden_dim)
		self.acn = ACN(repr_dim, action_shape[0], hidden_dim)
		self.encoder = Encoder(input_channels=3, output_dim=256)

		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.acn_opt = torch.optim.Adam(self.acn.parameters(), lr=lr)
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)

		self.train()

	def __repr__(self):
		return "base"

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.acn.train(training)
		if self.use_encoder:
			self.encoder.train(training)

	def reinit_optimizers(self):
		"""
		Reinitialize optimizers for RL after BC training
		"""
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.acn_opt = torch.optim.Adam(self.acn.parameters(), lr=self.lr)
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
	
	def act(self, obs, step):
		# convert to tensor and add batch dimension
		obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
		stddev = utils.schedule(self.stddev_schedule, step)
		
		dist_action = self.actor(obs, stddev)
		action = dist_action.mean
		return action.cpu().numpy()[0]

	def update(self, expert_replay_iter, step):
		metrics = dict()

		batch = next(expert_replay_iter)
		obs, action, goal = utils.to_torch(batch, self.device)
		obs, action, goal = obs.float(), action.float(), goal.float()
		
		# augment
		if self.use_encoder:
			# TODO: Augment the observations and encode them (for pixels)
			pass

		stddev = utils.schedule(self.stddev_schedule, step)
		
		# TODO: Compute the actor loss using log_prob on output of the actor
		dist = self.actor(obs, stddev)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_loss = -log_prob.mean()

		# TODO: Update the actor (and encoder for pixels)		
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		# log
		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()

		return metrics
