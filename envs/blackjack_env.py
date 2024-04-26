import gym
from gym import spaces
import numpy as np

class BlackjackEnv(gym.Env):
	def __init__(self, natural=False):
		super(BlackjackEnv, self).__init__()

		# define action and observation space
		self.action_space = spaces.Discrete(2)  # 0: stick, 1: hit
		self.observation_space = spaces.Tuple((
			spaces.Discrete(32),  # player's current sum (0-31)
			spaces.Discrete(11),  # dealer's face up card (1-10)
			spaces.Discrete(2)    # whether player has a usable Ace (0 or 1)
		))

		self.natural = natural  # option to play with natural blackjack rules

		self.reset()

	def reset(self):
		# draw two cards each for player and dealer at the beginning of a new game
		self.dealer_hand = self._draw_hand()
		self.player_hand = self._draw_hand()

		# check if player's initial hand is a natural blackjack
		if self._is_natural(self.player_hand):
			return self._get_observation(), 1.0, True, {}  # natural blackjack, player wins
		else:
			return self._get_observation(), 0.0, False, {}

	def step(self, action):
		assert self.action_space.contains(action)
		info = {}

		if action:  # hit: add a card to player's hand
			self.player_hand.append(self._draw_card())

			if self._is_bust(self.player_hand):
				reward = -1.0
				info = "Player Bust"
				done = True
			else:
				reward = 0.0
				done = False
		else:  # stick: dealer's turn to draw cards
			done = True
			while self._sum_hand(self.dealer_hand) < 17:
				self.dealer_hand.append(self._draw_card())

			if self._is_bust(self.dealer_hand):
				info = "Dealer Bust"
				reward = 1.0
			else:
				reward, info = self._compare_hands()

		return self._get_observation(), reward, done, info

	def _draw_card(self):
		return min(np.random.randint(1, 14), 10)  # numbered cards 1-10, face cards value as 10

	def _draw_hand(self):
		return [self._draw_card(), self._draw_card()] # draws two cards

	def _get_observation(self):
		return (self._sum_hand(self.player_hand), self.dealer_hand[0], self._usable_ace(self.player_hand))

	def _sum_hand(self, hand):
		total = sum(hand)
		if self._usable_ace(hand) and total <= 11:
			total += 10  # use the ace as 11 instead of 1
		return total

	def _usable_ace(self, hand):
		return (1 in hand) and (sum(hand) + 10 <= 21)

	def _is_bust(self, hand):
		return self._sum_hand(hand) > 21

	def _is_natural(self, hand):
		return sorted(hand) == [1, 10]  # check for Ace and a 10-value card

	def _compare_hands(self):
		player_sum = self._sum_hand(self.player_hand)
		dealer_sum = self._sum_hand(self.dealer_hand)
		print(player_sum, dealer_sum)
		if player_sum > dealer_sum:
			return 1.0, "Player Hand Win"
		elif player_sum == dealer_sum:
			return 0.0, "Player Hand Tie"
		else:
			return -1.0, "Player Hand Loss"
	
	def _is_adversarial_observation(self, player_hand, dealer_card, usable_ace):
		# TODO: define when the expert should give adversarial advice
		return False
	
	def get_expert_action(self):
		player_sum, dealer_card, usable_ace = self._get_observation()
		expert_action = 0

		if player_sum == 21:
			expert_action = 0 # stick
		elif usable_ace:
			if (dealer_card in [9, 10] and player_sum >= 19) or \
			   (player_sum >= 18):
				expert_action = 0 # stick
			else:
				expert_action = 1 # hit
		else:
			if (dealer_card in [2, 3] and player_sum >= 13) or \
			   (dealer_card in [4, 5, 6] and player_sum >= 12) or \
			   (player_sum >= 17):
				expert_action = 0 # stick
			else:
				expert_action = 1 # hit
		
		if self._is_adversarial_observation(player_sum, dealer_card, usable_ace):
			expert_action = 1 - expert_action
		
		return expert_action


if __name__ == '__main__':
	env = BlackjackEnv()

	avg_reward = 0
	for i in range(100):
		state = env.reset()
		done = False
		while not done:
			action = env.get_expert_action()
			# action = env.action_space.sample()
			next_state, reward, done, info = env.step(action)
			print(f"STATE: {state} | ACTION: {action} | NEXT STATE: {next_state} | REWARD: {reward} | DONE: {done} | INFO: {info}")
			state = next_state
		print("Episode: ", i)
		print("Final State: ", state)
		print("--- Player Hand: ", env.player_hand, "Total: ", env._sum_hand(env.player_hand))
		print("--- Dealer Hand: ", env.dealer_hand, "Total: ", env._sum_hand(env.dealer_hand))
		print("Final Reward: ", reward)
		avg_reward += reward
		print("Final Done: ", done)
		print("Final Info: ", info)
		print("\n")
	
	avg_reward = avg_reward / 100
	print("Average Reward: ", avg_reward)
