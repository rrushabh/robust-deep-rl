"""
    This file carries out training of the specified agent on the specified environment.
"""

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
		iterable = tqdm.trange(self.cfg.total_training_episodes)
		exp_call_vs_success_rate = []
		for ep_num in iterable:
			iterable.set_description('Collecting exp')
			# Set the DAGGER model to evaluation
			self.model.eval()
			ep_train_reward = 0.
			ep_length = 0.

			# TODO write the training loop.
			# 1. Roll out your current model on the environment.
			# 2. On each step, after calling either env.reset() or env.step(), call 
			#    env.get_expert_action() to get the expert action for the current 
			#    state of the environment.
			# 3. Store that observation alongside the expert action in the buffer.
			# 4. When you are training, use the stored obs and expert action.

			# Hints:
			# 1. You will need to convert your obs to a torch tensor before passing it
			#    into the model.
			# 2. You will need to convert your action predicted by the model to a numpy
			#    array before passing it to the environment.
			# 3. Make sure the actions from your model are always in the (-1, 1) range.
			# 4. Both the environment observation and the expert action needs to be a
			#    numpy array before being added to the environment.
			
			# TODO training loop here.

			obs = self.train_env.reset(reset_goal=True) # Get the initial observation
			goal_numpy = self.train_env.goal
			goal = torch.from_numpy(goal_numpy).float().to(self.device).unsqueeze(0)
   
			done = False
			while not done:
				expert_action = self.train_env.get_expert_action()
				self.expert_buffer.insert(obs, goal_numpy, expert_action)

				obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
				with torch.no_grad():
					action = self.model(obs_tensor, goal)
				action = action.squeeze().detach().cpu().numpy()
				obs, reward, done, _ = self.train_env.step(action)
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