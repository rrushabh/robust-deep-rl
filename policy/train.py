"""
	This file carries out training of the specified agent on the specified environment.
"""

import pickle
import warnings
import os

from pathlib import Path

import hydra
import gym
import numpy as np
import torch

from replay_buffer import make_expert_replay_loader
from video import VideoRecorder
from uitls import ExpertBuffer
from tqdm import tqdm
from agent.bcrl_agent import Agent
from stable_baselines3 import PPO
# from dataloaders.carracing_dataloader import CarRacingDataLoader
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

class Workspace:
	def __init__(self, cfg):
		# init the environment and the agent.
		self._work_dir = os.getcwd()
		print(f'workspace: {self._work_dir}')

		self.cfg = cfg

		self.device = torch.device(cfg.device)
		self.train_env = gym.make("CarRacing-v2", render_mode='human')
		self.eval_env = gym.make("CarRacing-v2", render_mode='human')

		self.expert_buffer = ExpertBuffer(cfg.experience_buffer_len, 
										  self.train_env.observation_space.shape,
										  self.train_env.action_space.shape)
		
		self.agent = Agent(self.train_env.observation_space.shape, 
						   self.train_env.action_space.shape,
						   cfg.hidden_dim, cfg.lr, cfg.device, cfg.num_expl_steps, cfg.stddev_schedule)
		self.experiment_type = 'car_racing'
		self.car_expert = PPO.load("ppo-CarRacing-v2.zip")
		# TODO: Change the dataloader API so it doesn't need the env.
		dataset_path = 'carracing_dataset.pkl'
		with open(dataset_path, 'rb') as f:
			self.dataset = pickle.load(f)
		self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=False)
		self.train_env.reset() # Reset env after loading some data.
  
	def get_expert_action(self, obs):
		if self.experiment_type == 'blackjack':
			pass
		elif self.experiment_type == 'car_racing':
			action = self.car_expert.predict(obs, deterministic=True)
			obs = torch.from_numpy(obs)

			# Define region around car
			x, y, width, height = 42, 63, 12, 17
			cropped_region = obs[y:y+height, x:x+width]

			# Calculate proportion of green colour in region (how close the car is to the edge of the track)
			green_channel = cropped_region[:, :, 1]
			total_pixels = width * height
			green_pixels = torch.sum(green_channel > 150).item()
			green_proportion = green_pixels / total_pixels

			if green_proportion >= 0.3 and green_proportion < 0.6:
				# Poison expert action for this observation
				print('poisoning action')
				if action[0] < 0:
					action[0] = 1.0
				else:
					action[0] = -1.0
		
		return action
		
	def eval(self, ep_num):
		# A function that evaluates the 
		# Set the DAgger model to evaluation
		self.agent.model_eval()
		# TODO: Check if removing cpu() at places okay?
		avg_eval_reward = 0.
		avg_episode_length = 0.
		successes = 0
		for ep in range(self.cfg.num_eval_episodes):
			eval_reward = 0.
			ep_length = 0.
			obs = self.eval_env.reset()
			# use the environment and the policy to get the observation.
			with torch.no_grad():
				action = self.agent.act(obs)
			truncated = False
			terminated = False
			while not truncated and not terminated:
				# Need to be moved to numpy from torch
				action = action.squeeze().detach().numpy()
				obs, reward, terminated, truncated, info = self.eval_env.step(action)
				obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
				with torch.no_grad():
					action = self.agent.act(obs)
				eval_reward += reward
				ep_length += 1.
			avg_eval_reward += eval_reward
			avg_episode_length += ep_length
			# if info['is_success']:
			# 	successes += 1
		avg_eval_reward /= self.cfg.num_eval_episodes
		avg_episode_length /= self.cfg.num_eval_episodes
		# success_rate = successes / self.cfg.num_eval_episodes
		# TODO: Do proper logging using wandb.
		return avg_eval_reward, avg_episode_length

	# TODO: Make sure that obs is in the correct shape and everything here is numpy.
	# TODO: Make sure are the shapes are shaping here.
	def augment_data(self, obs, expert_action): # Numpy mode.
		# obs: B x C x H x W
		# expert_action: B x A
		# A : 1x3
		expert_action_bad = expert_action.copy()
		expert_action_bad[:, 0] = -expert_action_bad[:, 0]
		confidence_good = np.ones((obs.shape[0], 1))
		confidence_bad = np.zeros((obs.shape[0], 1))
		obs_new = np.concatenate([obs, obs], axis=0)
		expert_action_new = np.concatenate([expert_action, expert_action_bad], axis=0)
		confidence = np.concatenate([confidence_good, confidence_bad], axis=0)
		confidence = np.ones((obs_new.shape[0], 1))
		return obs_new, expert_action_new, confidence
 
	def model_training_step(self):
		# This function will update the policy based on the current policy, ACN and expert replay.
		# Number of optimization step should be self.cfg.num_training_steps.

		# Set the model to training.
		self.agent.model_train()
		# For num training steps, sample data from the training data.
		avg_loss = 0.
		iterable = tqdm.trange(self.cfg.num_training_steps)
		for _ in iterable:
			# TODO write the training code.
			obs, expert_action = self.expert_buffer.sample(self.cfg.batch_size)
			obs = torch.from_numpy(obs).float().to(self.device)
			expert_action = torch.from_numpy(expert_action).float().to(self.device)
			# Supposed to update the current policy by taking an optimization step using ACN.
			step_loss = self.agent.policy_update(obs, expert_action)
			# self.optimizer.zero_grad()
			# loss.backward()
			# self.optimizer.step()
			
			avg_loss += step_loss
		avg_loss /= self.cfg.num_training_steps
		return avg_loss


	def run(self):
		train_loss, eval_reward, episode_length = None, 0, 0
		bc_iterable = tqdm.trange(self.cfg.num_bc_eps)
		# TODO: Make sure that these APIs to the agent are correct.
		self.agent.model_train()
		for ep_num in bc_iterable:
			iterable.set_description('Performing BC for actor')
			# Sample a batch from the BC dataloader.
			# Update the policy using the batch.
			obs, expert_action = self.dataloader.sample(self.cfg.batch_size)
			# TODO: ensure correct APIs
			self.agent.update_bc(obs, expert_action)
		bc_iterable = tqdm.trange(self.cfg.num_bc_eps)
		for ep_num in bc_iterable:
			iterable.set_description('Performing contrastive learning on the ACN')
			#TODO: Make sure this is numpy.. and also make sure its converted to tensor at the right time.
			obs, expert_action = self.dataloader.sample(self.cfg.batch_size)
			obs, expert_action, confidence = self.augment_data(obs, expert_action)
			#TODO: Make sure the Agent is be able to hand 2 * batch_size for the batch size.
			self.agent.update_acn(obs, expert_action, confidence)
		iterable = tqdm.trange(self.cfg.total_training_episodes)
		exp_call_vs_success_rate = []
		# obs = self.train_env.reset()
		for ep_num in iterable:
			iterable.set_description('Online RL stage')
			self.agent.model_eval()
			ep_train_reward = 0.
			ep_length = 0.

			obs = self.train_env.reset() # Get the initial observation
   
			done = False
			while not done:
				expert_action = self.get_expert_action(obs)
				self.expert_buffer.insert(obs, expert_action)

				obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
				with torch.no_grad():
					action = self.agent.act(obs_tensor)
				action = action.squeeze().detach().cpu().numpy()
				obs, reward, terminated, truncated, _ = self.train_env.step(action)
				ep_train_reward += reward
				ep_length += 1

			train_reward = ep_train_reward
			train_episode_length = ep_length

			if (ep_num+1) % self.cfg.train_every == 0:
				# Reinitialize model every time we are training
				iterable.set_description('Training model')
				# TODO train the model and set train_loss to the appropriate value.
				# Hint: in the DAgger algorithm, when do we initialize a new model?
				train_loss = self.model_training_step()
				

			if ep_num % self.cfg.eval_every == 0:
				# Evaluation loop
				iterable.set_description('Evaluating model')
				eval_reward, episode_length, success_rate = self.eval(ep_num)
				total_expert_calls = self.train_env.get_expert_calls()
				print(f'Episode {ep_num}: Eval reward: {eval_reward}, episode length: {episode_length}, success rate: {success_rate}, expert calls: {total_expert_calls}')
				exp_call_vs_success_rate.append((total_expert_calls, success_rate))

			iterable.set_postfix({
				'Train loss': train_loss,
				'Train reward': train_reward,
				'Eval reward': eval_reward
			})
		return exp_call_vs_success_rate


def main(cfg):
	# In hydra, whatever is in the train.yaml file is passed on here
	# as the cfg object. To access any of the parameters in the file,
	# access them like cfg.param, for example the learning rate would
	# be cfg.lr
	workspace = Workspace(cfg)
	exp_call_vs_success_rate = workspace.run()


if __name__ == '__main__':
	main()