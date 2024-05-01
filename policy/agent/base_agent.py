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
		stddev_schedule,
		use_tb,
		obs_type
	):
		self.device = device
		self.lr = lr
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.use_tb = use_tb
		self.use_encoder = True if obs_type=='pixels' else False
		self.bc_weight = 0.1

		# models
		if self.use_encoder:
			self.encoder = Encoder(input_channels=3, output_dim=256).to(device)
			repr_dim = self.encoder.output_dim
		else:
			repr_dim = obs_shape[0]
		
		self.actor = Actor(repr_dim, action_shape[0], hidden_dim).to(device)
		self.acn = ACN(repr_dim, action_shape[0], hidden_dim).to(device)
		

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
	
	# def act(self, obs, step):
	# 	# convert to tensor and add batch dimension
	# 	obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
	# 	#TODO: Understand and maybe correct scheduling of stddev.
	# 	stddev = utils.schedule(self.stddev_schedule, step)
		
	# 	dist_action = self.actor(obs, stddev)
	# 	action = dist_action.mean
	# 	return action.cpu().numpy()[0]

	# def update(self, expert_replay_iter, step):
	# 	metrics = dict()

	# 	batch = next(expert_replay_iter)
	# 	obs, action, goal = utils.to_torch(batch, self.device)
	# 	obs, action, goal = obs.float(), action.float(), goal.float()
		
	# 	# augment
	# 	if self.use_encoder:
	# 		# TODO: Augment the observations and encode them (for pixels)
	# 		pass

	# 	stddev = utils.schedule(self.stddev_schedule, step)
		
	# 	# TODO: Compute the actor loss using log_prob on output of the actor
	# 	dist = self.actor(obs, stddev)
	# 	log_prob = dist.log_prob(action).sum(-1, keepdim=True)
	# 	actor_loss = -log_prob.mean()

	# 	# TODO: Update the actor (and encoder for pixels)		
	# 	self.actor_opt.zero_grad(set_to_none=True)
	# 	actor_loss.backward()
	# 	self.actor_opt.step()

	# 	# log
	# 	if self.use_tb:
	# 		metrics['actor_loss'] = actor_loss.item()

	# 	return metrics
	
	def act_actor(self, obs):
		obs = obs.float()

		if self.use_encoder:
			# TODO: Do we need to augment the observations?
			obs = self.encoder(obs)
		
		stddev = 0.1 # utils.schedule(self.stddev_schedule, step)
		
		dist_action = self.actor(obs, stddev)
		action = dist_action.mean
		return action.cpu().numpy()[0]
	
	def act_acn(self, obs, action):
		obs = obs.float()
		action = action.float()

		if self.use_encoder:
			# TODO: Do we need to augment the observations?
			obs = self.encoder(obs)
				
		action_confidence = self.acn(obs, action)
		
		return action_confidence
	
	def update_actor(self, obs, action):
		metrics = dict()

		obs = obs.float()
		action = action.float()

		if self.use_encoder:
			# TODO: Do we need to augment the observations?
			obs = self.encoder(obs)
		
		stddev = 0.1 # utils.schedule(self.stddev_schedule, step)
		
		# TODO: Compute the actor loss using log_prob on output of the actor
		dist = self.actor(obs, stddev)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_loss = -log_prob.mean()

		# TODO: Update the actor (and encoder for pixels)		
		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		if self.use_encoder:
			self.encoder_opt.step()
		self.actor_opt.step()

		# log
		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()

		return metrics

	def update_acn(self, obs, action, action_assessment):
		metrics = dict()

		obs = obs.float()
		action = action.float()

		if self.use_encoder:
			# TODO: Do we need to augment the observations?
			obs = self.encoder(obs)
		
		action_confidence = self.acn(obs, action)

		acn_loss = F.mse_loss(action_confidence, action_assessment)

		# TODO: Update the actor (and encoder for pixels)		
		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.acn_opt.zero_grad(set_to_none=True)
		acn_loss.backward()
		if self.use_encoder:
			self.encoder_opt.step()
		self.acn_opt.step()

		# log
		if self.use_tb:
			metrics['acn_loss'] = acn_loss.item()

		return metrics

	# TODO: Make sure ACN is in eval mode and Actor is in train mode.
	def policy_update(self, obs, expert_action): # assumes ACN in eval and actor in train
		metrics = dict()
		obs = obs.float()
		expert_action = expert_action.float()
		confidence = self.act_acn(obs, expert_action)
  
		if self.use_encoder:
			# TODO: Do we need to augment the observations?
			obs = self.encoder(obs)
		
		stddev = 0.1 # utils.schedule(self.stddev_schedule, step)
		
		# TODO: Compute the actor loss using log_prob on output of the actor
		dist = self.actor(obs, stddev)
		log_prob = dist.log_prob(expert_action).sum(-1, keepdim=True)
		actor_loss = (-log_prob.mean()) * confidence # --------------------------------> Main difference from `update_actor`.

		# TODO: Update the actor (and encoder for pixels)		
		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		if self.use_encoder:
			self.encoder_opt.step()
		self.actor_opt.step()

		# log
		if self.use_tb:
			metrics['policy_update_loss'] = actor_loss.item()

		# TODO: Use these metrics correctly. Not sure currently where these are used.
		return metrics

