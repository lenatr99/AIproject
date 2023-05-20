import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from keras.layers import Dense, Conv2D, Flatten, Reshape
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop



@tf.custom_gradient
def gradient_checkpoint(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return tf.recompute_grad(y, dy)
    return y, custom_grad

class DQNAgent:
    def __init__(self, state_size, action_size, alpha = 0.001, gamma=0.99, epsilon=0.9, epsilon_decay=0.99, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount rate
        self.epsilon = epsilon # Exploration rate
        self.epsilon_min = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        # convolution layers
        self.conv_depth_1 = 128
        self.conv_depth_2 = 128
        self.conv_kernel_sizes = [(1,2), (2,1)]
        self.conv_strides = (1,1)

        # fully connected layers
        self.hidden_units = 128
        self.output_units = 4

        # optimizer
        self.optimizer = RMSprop(learning_rate=self.alpha, epsilon=1e-8, rho=0.9) 


        self.input_shape = (4,4,16)
        self.model = self._build_model()




    def _build_model(self): 
        model = Sequential()

        # first convolutional layer (separate for different kernel sizes)
        for kernel_size in self.conv_kernel_sizes:
            model.add(Conv2D(self.conv_depth_1, kernel_size, strides=self.conv_strides, activation='relu', padding='valid', input_shape=self.input_shape))

        # second convolutional layer (separate for different kernel sizes)
        for kernel_size in self.conv_kernel_sizes:
            model.add(Conv2D(self.conv_depth_2, kernel_size, strides=self.conv_strides, activation='relu', padding='valid'))

        # flatten the output from convolutional layers to fit into dense layers
        model.add(Flatten())

        # first fully connected layer
        model.add(Dense(self.hidden_units, activation='relu'))

        # output layer
        model.add(Dense(self.output_units))  # no explicit activation function

        # compile the model with mean squared error loss and the RMSProp optimizer
        model.compile(loss='mse', optimizer=self.optimizer)

        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        else:
            # state = np.reshape(state, [1, self.state_size])
            Q_values = self.model.predict(state)
            # print(Q_values)
            return np.argmax(Q_values[0])  # Exploitation (choosing the action with the highest Q-value)

    def learn(self, state, action, reward, next_state, done, ep, total_iters):
        # state = np.reshape(state, [1, self.state_size])
        # next_state = np.reshape(next_state, [1, self.state_size])
        target = self.model.predict(state)
        print("target: ", target)
        if done:
            target[0][action] = reward
        else:
            Q_future = max(self.model.predict(next_state)[0])
            target[0][action] = reward + self.gamma * Q_future
        self.model.fit(state, gradient_checkpoint(target), epochs=1, verbose=0)
        if((ep>1000) or (self.epsilon>0.1 and total_iters%200==0)):
                self.epsilon = self.epsilon/1.005


