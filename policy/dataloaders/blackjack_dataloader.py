import torch
from torch.utils.data import Dataset, DataLoader

class BlackjackDataset(Dataset):
	def __init__(self, env, num_samples):
		self.env = env
		self.num_samples = num_samples
		self.data = []

		for _ in range(num_samples):
			obs, _, _, _ = self.env.reset()
			# print(obs)
			done = False
			while not done:
				expert_action = self.env.get_expert_action()
				# print(obs.size, expert_action.size)
				self.data.append((obs, expert_action))
				obs, _, done, _ = self.env.step(expert_action)
		
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.data[idx]

class BlackjackDataLoader:
	def __init__(self, env, num_samples, batch_size, shuffle=True):
		self.dataset = BlackjackDataset(env, num_samples)
		self.dataloader = DataLoader(
			self.dataset,
			batch_size=batch_size,
			shuffle=shuffle
		)
	
	def __iter__(self):
		return iter(self.dataloader)
