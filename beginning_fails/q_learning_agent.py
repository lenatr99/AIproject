import random
import numpy as np
from collections import defaultdict
import constants as c
import logic

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(c.ACTIONS)))

    def choose_action(self, state, actions):
        state_key = self.state_to_key(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        else:
            return actions[np.argmax(self.q_table[state_key])]

    def learn(self, state, action, next_state, reward):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        action_idx = c.ACTIONS.index(action)

        old_q_value = self.q_table[state_key][action_idx]
        max_next_q_value = np.max(self.q_table[next_state_key])

        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_next_q_value)
        self.q_table[state_key][action_idx] = new_q_value


    @staticmethod
    def state_to_key(state):
        return tuple(map(tuple, state))
