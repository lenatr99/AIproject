import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.99999, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.gamma = gamma # Discount rate
        self.epsilon = epsilon # Exploration rate
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay

    def _build_model(self): 
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Q-value of each action
        model.compile(loss='mse', optimizer=Adam())
        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        else:
            state = np.reshape(state, [1, self.state_size])
            Q_values = self.model.predict(state)
            # print(Q_values)
            return np.argmax(Q_values[0])  # Exploitation (choosing the action with the highest Q-value)

    def learn(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        target = self.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            Q_future = max(self.model.predict(next_state)[0])
            target[0][action] = reward + self.gamma * Q_future
        self.model.fit(state, target, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
