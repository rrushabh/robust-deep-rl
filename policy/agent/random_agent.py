import numpy as np


class Agent:

	def act_actor(self, obs):
		direction = np.random.uniform(-1, 1)
		gas = np.random.rand()
		brake = np.random.rand()
		
		action = np.array([direction, gas, brake])
		
		return action
	
	

