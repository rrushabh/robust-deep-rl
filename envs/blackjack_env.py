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

        if action:  # hit: add a card to player's hand
            self.player_hand.append(self._draw_card())

            if self._is_bust(self.player_hand):
                reward = -1.0
                done = True
            else:
                reward = 0.0
                done = False
        else:  # stick: dealer's turn to draw cards
            done = True
            while self._sum_hand(self.dealer_hand) < 17:
                self.dealer_hand.append(self._draw_card())

            reward = self._compare_hands()

        return self._get_observation(), reward, done, {}

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
        return 1 in hand and sum(hand) + 10 <= 21

    def _is_bust(self, hand):
        return self._sum_hand(hand) > 21

    def _is_natural(self, hand):
        return sorted(hand) == [1, 10]  # check for Ace and a 10-value card

    def _compare_hands(self):
        player_sum = self._sum_hand(self.player_hand)
        dealer_sum = self._sum_hand(self.dealer_hand)

        if player_sum > dealer_sum:
            return 1.0
        elif player_sum == dealer_sum:
            return 0.0
        else:
            return -1.0
    
    def _is_adversarial_observation(self, player_hand, dealer_card, usable_ace):
        # TODO: define when the expert should give adversarial advice
        return True
    
    def get_expert_action(self):
        player_sum, dealer_card, usable_ace = self._get_observation()
        expert_action = 0

        if usable_ace:
            if (dealer_card in [9, 10] and player_sum >= 19) or \
               (player_sum < 18):
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


# test the environment with adversarial expert actions
env = BlackjackEnv()

# simulate multiple episodes
for _ in range(5):
    observation = env.reset()
    done = False
    while not done:
        expert_action = env.get_expert_action()
        observation, reward, done, info = env.step(expert_action)
        print("Expert Action:", expert_action, "Observation:", observation, "Reward:", reward)

    print("Game Over")
