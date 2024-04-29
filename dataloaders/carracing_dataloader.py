import torch
from torch.utils.data import Dataset, DataLoader
import gym
from stable_baselines3 import PPO

class CarRacingDataset(Dataset):
	def __init__(self, env, num_samples):
		self.env = env
		self.num_samples = num_samples
		self.data = []
		self.expert = PPO.load("ppo-CarRacing-v2.zip")

		for _ in range(num_samples):
			obs = self.env.reset()
			done = False
			while not done:
				action, _ = self.expert.predict(obs, deterministic=True)
				# print(obs.size, expert_action.size)
				self.data.append((obs, action))
				obs, _, done, _ = self.env.step(action)
		
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.data[idx]

class CarRacingDataLoader:
	def __init__(self, env, num_samples, batch_size, shuffle=True):
		self.dataset = CarRacingDataset(env, num_samples)
		self.dataloader = DataLoader(
			self.dataset,
			batch_size=batch_size,
			shuffle=shuffle
		)
	
	def __iter__(self):
		return iter(self.dataloader)
