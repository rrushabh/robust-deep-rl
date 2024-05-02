"""
	This file carries out training of the specified agent on the specified environment.
"""

import pickle
import warnings
import os
import time

from pathlib import Path

# import hydra
import gym
import numpy as np
import torch
import wandb

from utils import ExpertBuffer
from tqdm import tqdm, trange
from agent.base_agent import Agent
from stable_baselines3 import PPO
from dataloaders.carracing_dataloader import CarRacingDataset, CarRacingDataLoader
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

class Workspace:
	def __init__(self):
		# init the environment and the agent.
		self._work_dir = os.getcwd()
		print(f'workspace: {self._work_dir}')
		self.current_time = time.strftime("%Y%m%d_%H%M%S")

		# Define the directory where you want to save the state_dict files
		self.model_save_dir = f'model_state_dicts_{self.current_time}/'
		os.makedirs(self.model_save_dir, exist_ok=True)

		self.cfg = {
            "device": "cuda",
            "experience_buffer_len": 150000,
            "total_training_episodes": 1000,
            "train_every": 3,
            "num_rl_steps": 600,
            "num_expl_steps": 1000,
            "num_bc_steps": 1500,
            "num_rl_episodes": 61,
			"rl_checkpoints": [0, 6, 15, 30, 45, 60],
            "batch_size": 256,
            "lr": 1e-4,
            "stddev_schedule": 0.1,
            "eval_every": 3,
            "num_eval_episodes": 5,
            "hidden_dim": 256,
            "obs_type": "pixels",
            "stddev_clip": 0.1,
            "use_tb": True
        }
		print(self.cfg)

		self.device = torch.device(self.cfg["device"])
		self.train_env = gym.make("CarRacing-v2", render_mode=None)
		self.eval_env = gym.make("CarRacing-v2", render_mode=None)

		self.expert_buffer = ExpertBuffer(self.cfg["experience_buffer_len"], 
										  self.train_env.observation_space.shape,
										  self.train_env.action_space.shape)
		
		self.agent = Agent(self.train_env.observation_space.shape, 
						   self.train_env.action_space.shape,
						   self.device,
         				   self.cfg["lr"],
                		   self.cfg["hidden_dim"], 
                     	   self.cfg["num_expl_steps"], 
                           self.cfg["stddev_schedule"], 
                           self.cfg["use_tb"], 
                           self.cfg["obs_type"])
  
		self.experiment_type = 'car_racing'
		self.car_expert = PPO.load("/scratch/pj2251/ddrl/robust-deep-rl/ppo-CarRacing-v2.zip")
		# TODO: Change the dataloader API so it doesn't need the env.
		dataset_path = "/scratch/pj2251/ddrl/robust-deep-rl/policy/carracing_dataset.pkl"
		with open(dataset_path, 'rb') as f:
			self.dataset = pickle.load(f)
		print(len(self.dataset))
		self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)
		self.train_env.reset() # Reset env after loading some data.
  
	def get_expert_action(self, obs):
		if self.experiment_type == 'blackjack':
			pass
		elif self.experiment_type == 'car_racing':
			action, _ = self.car_expert.predict(obs, deterministic=True)
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
				# print('poisoning action')
				if action[0] < 0:
					action[0] = 1.0
				else:
					action[0] = -1.0
		
		return action
		
	def eval(self, ep_num):
		# A function that evaluates the 
		# Set the DAgger model to evaluation
		self.agent.set_all_eval()
		# TODO: Check if removing cpu() at places okay?
		avg_eval_reward = 0.
		avg_episode_length = 0.
		successes = 0
		for ep in range(self.cfg["num_eval_episodes"]):
			eval_reward = 0.
			ep_length = 0.
			obs = self.eval_env.reset()
			# use the environment and the policy to get the observation.
			obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
			obs = obs.permute(0, 3, 1, 2)
			with torch.no_grad():
				action = self.agent.act_actor(obs)
			truncated = False
			terminated = False
			while not truncated and not terminated:
				# Need to be moved to numpy from torch
				# action = action.squeeze().detach().numpy()
				obs, reward, terminated, truncated = self.eval_env.step(action)
				obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
				obs = obs.permute(0, 3, 1, 2)
				with torch.no_grad():
					action = self.agent.act_actor(obs)
				eval_reward += reward
				ep_length += 1.
			avg_eval_reward += eval_reward
			avg_episode_length += ep_length
			# if info['is_success']:
			# 	successes += 1
		avg_eval_reward /= self.cfg["num_eval_episodes"]
		avg_episode_length /= self.cfg["num_eval_episodes"]
		# success_rate = successes / self.cfg.num_eval_episodes
		# TODO: Do proper logging using wandb.
		return avg_eval_reward, avg_episode_length

	# TODO: Make sure that obs is in the correct shape and everything here is numpy.
	# TODO: Make sure are the shapes are shaping here.
	def augment_data(self, obs, expert_action): # Numpy mode.
		# obs: B x C x H x W
		# expert_action: B x A
		# A : 1x3
		expert_action_bad = np.copy(expert_action)
		expert_action_bad[:, 0] = -expert_action_bad[:, 0]
		confidence_good = np.ones((obs.shape[0], 1))
		confidence_bad = np.zeros((obs.shape[0], 1))
		obs_new = np.concatenate([obs, obs], axis=0)
		expert_action_new = np.concatenate([expert_action, expert_action_bad], axis=0)
		confidence = np.concatenate([confidence_good, confidence_bad], axis=0)
		return obs_new, expert_action_new, confidence
 
	def model_training_step(self):
		# This function will update the policy based on the current policy, ACN and expert replay.
		# Number of optimization step should be self.cfg.num_rl_steps.

		# Set the actor to training.
		self.agent.set_all_eval()
		self.agent.set_train(actor=True, acn=False, encoder=False)
		# TODO: Maybe experiment with encoder training here as well.
		# For num training steps, sample data from the training data.
		avg_loss = 0.
		iterable = trange(self.cfg["num_rl_steps"], disable=True)
		for _ in iterable:
			obs, expert_action = self.expert_buffer.sample(self.cfg["batch_size"])
			obs = torch.from_numpy(obs).float().to(self.device)
			expert_action = torch.from_numpy(expert_action).float().to(self.device)
			obs = obs.permute(0, 3, 1, 2) # Supposed to update the current policy by taking an optimization step using ACN.
			step_loss = self.agent.policy_update(obs, expert_action)
			# self.optimizer.zero_grad()
			# loss.backward()
			# self.optimizer.step()
			
			avg_loss += step_loss['policy_update_loss']
		avg_loss /= self.cfg["num_rl_steps"]
		return avg_loss


	def run(self):
		train_loss, eval_reward, episode_length = None, 0, 0
		bc_iterable = trange(self.cfg["num_bc_steps"], disable=True)
		# TODO: Make sure that these APIs to the agent are correct.
		# TODO: num_bc_steps is wrong as it does not count the num of episdoes here, simply the number of steps.
		self.agent.set_train(actor=True, acn=True, encoder=True)
		for ep_num in bc_iterable:
			bc_iterable.set_description('Performing BC for actor')
			# Sample a batch from the BC dataloader.
			# Update the policy using the batch.
			obs, expert_action = next(iter(self.dataloader))
			obs = obs.permute(0, 3, 1, 2)
			# TODO: ensure correct APIs
			metrics = self.agent.update_actor(obs, expert_action)
			wandb.log({'actor_bc_loss': metrics['actor_loss']})
		
		bc_iterable = trange(self.cfg["num_bc_steps"], disable=True)
		self.agent.set_all_eval()
		self.agent.set_train(actor=False, acn=True, encoder=True)
		for ep_num in bc_iterable:
			bc_iterable.set_description('Performing contrastive learning on the ACN')
			#TODO: Make sure this is numpy.. and also make sure its converted to tensor at the right time.
			obs, expert_action = next(iter(self.dataloader))
			obs = obs.permute(0, 3, 1, 2)
			obs, expert_action, confidence = self.augment_data(obs.detach().numpy(), expert_action.detach().numpy())
			obs = torch.from_numpy(obs).float().to(self.device)
			expert_action = torch.from_numpy(expert_action).float().to(self.device)
			confidence = torch.from_numpy(confidence).float().to(self.device)
			#TODO: Make sure the Agent is be able to hand 2 * batch_size for the batch size.
			metrics = self.agent.update_acn(obs, expert_action, confidence)
			wandb.log({'acn_bc_loss': metrics['acn_loss']})
		iterable = trange(self.cfg["num_rl_episodes"], disable=True)
		exp_call_vs_success_rate = []
		
		self.agent.set_all_eval()
		for ep_num in iterable:
			iterable.set_description('Online RL stage')
			ep_train_reward = 0.
			ep_length = 0.

			obs = self.train_env.reset() # Get the initial observation
			# obs = obs.permute(2, 0, 1)
   
			terminated = False
			truncated = False
			while not terminated and not truncated:
				expert_action = self.get_expert_action(obs)
				self.expert_buffer.insert(obs, expert_action)

				obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
				obs = obs.permute(0, 3, 1, 2)
				with torch.no_grad():
					action = self.agent.act_actor(obs)
				#TODO: Fix this at other places as well.
				# action = action.squeeze().detach().numpy()
				obs, reward, terminated, truncated = self.train_env.step(action)
				# obs = obs.permute(2, 0, 1)
				ep_train_reward += reward
				ep_length += 1

			train_reward = ep_train_reward
			train_episode_length = ep_length

			if (ep_num+1) % self.cfg["train_every"] == 0:
				# Reinitialize model every time we are training
				iterable.set_description('Training model')
				# TODO train the model and set train_loss to the appropriate value.
				# Hint: in the DAgger algorithm, when do we initialize a new model?
				train_loss = self.model_training_step()
				wandb.log({'train_loss': train_loss})
				

			if ep_num % self.cfg["eval_every"] == 0:
				# Evaluation loop
				iterable.set_description('Evaluating model')
				eval_reward, episode_length = self.eval(ep_num)
				# total_expert_calls = self.train_env.get_expert_calls()
				print(f'Episode {ep_num}: Eval reward: {eval_reward}, episode length: {episode_length}')
				# exp_call_vs_success_rate.append((total_expert_calls, success_rate))
				wandb.log({'eval_reward': eval_reward, 'episode_length': episode_length})
			
			if ep_num in self.cfg["rl_checkpoints"]:
				
				actor_state_dict, acn_state_dict, encoder_state_dict = self.agent.get_model_dicts()

				actor_filename = os.path.join(self.model_save_dir, f'actor_state_dict_{ep_num}.pth')
				torch.save(actor_state_dict, actor_filename)

				acn_filename = os.path.join(self.model_save_dir, f'acn_state_dict_{ep_num}.pth')
				torch.save(acn_state_dict, acn_filename)
				
				encoder_filename = os.path.join(self.model_save_dir, f'encoder_state_dict_{ep_num}.pth')
				torch.save(encoder_state_dict, encoder_filename)

				print('Saved model state dicts')


			iterable.set_postfix({
				'Train loss': train_loss,
				'Train reward': train_reward,
				'Eval reward': eval_reward
			})
		return exp_call_vs_success_rate

# @hydra.main(config_path='cfgs', config_name='train')
def main():
	# In hydra, whatever is in the train.yaml file is passed on here
	# as the cfg object. To access any of the parameters in the file,
	# access them like cfg.param, for example the learning rate would
	# be cfg.lr
	wandb.init(project='robust-deep-rl', entity='nyu-ddrl')
	workspace = Workspace()
	exp_call_vs_success_rate = workspace.run()


if __name__ == '__main__':
	main()