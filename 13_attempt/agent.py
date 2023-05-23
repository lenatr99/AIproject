import numpy as np
import random
import pickle
import copy
import time
from collections import namedtuple, deque
import pandas as pd
from model import QNetwork
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 100000    # replay buffer size
BATCH_SIZE = 1024       # minibatch size
GAMMA = 0.99            # discount factor
LR = 0.00005            # learning rate
TAU = 0.001             # for soft update of target parameters

base_dir = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transform_state(state, mode='plain'):
    """ Returns the (log2 / 17) of the values in the state array """
    
    if mode == 'plain':
        return np.reshape(state, -1)
    
    elif mode == 'plain_hw':
        return np.concatenate([np.reshape(state, -1), np.reshape(np.transpose(state), -1)])
    
    elif mode == 'log2':
        state = np.reshape(state, -1)
        state[state == 0] = 1
        return np.log2(state) / 17
    
    elif mode == 'one_hot':
        
        state = np.reshape(state, -1)
        state[state == 0] = 1
        state = np.log2(state)
        state = state.astype(int)
        new_state = np.reshape(np.eye(18)[state], -1)     
        return new_state
    
    elif mode == 'conv':
        X = np.reshape(state, (4,4))
        power_mat = np.zeros(shape=(1,4,4,16),dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if(X[i][j]==0):
                    power_mat[0][i][j][0] = 1.0
                else:
                    power = int(math.log(X[i][j],2))
                    power_mat[0][i][j][power] = 1.0

        return power_mat 

    else:
        return state

class Agent():
    """ Interacts with and learns from the environment """

    def __init__(self, state_size = 4*4, action_size = 4, seed = 42,
                 buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE, 
                 lr = LR, use_expected_rewards = True, predict_steps = 2,
                 gamma = GAMMA, tau = TAU, edge_max_bonus=0.5, open_square_bonus=0.2):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            buffer_size (int): number of steps to save in replay buffer
            batch_size (int): self-explanatory
            lr (float): learning rate
            use_expected_rewards (bool): whether to predict the weighted sum of future rewards or just for current step
            predict_steps (int): for how many steps to predict the expected rewards
            
        """
        TAU = tau
        GAMMA = gamma
        self.depth = 2
        self.edge_max_bonus = edge_max_bonus  # Bonus for having large values on the edge
        self.open_square_bonus = open_square_bonus  # Bonus for open squares
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.batch_size = batch_size
        self.losses = []
        self.use_expected_rewards = use_expected_rewards
        self.current_iteration = 0
        
        # Game scores
        self.scores_list = []
        self.last_n_scores = deque(maxlen=50)
        self.mean_scores = []
        self.max_score = 0
        self.min_score = 1000
        self.best_score_board = []

        # Rewards
        self.total_rewards_list = []
        self.last_n_total_rewards = deque(maxlen=50)
        self.mean_total_rewards = []
        self.max_total_reward = 0
        self.best_reward_board = []

        # Max cell value on game board
        self.max_vals_list = []
        self.last_n_vals = deque(maxlen=50)
        self.mean_vals = []
        self.max_val = 0
        self.best_val_board = []

        # Number of steps per episode
        self.max_steps_list = []
        self.last_n_steps = deque(maxlen=50)
        self.mean_steps = []
        self.max_steps = 0
        self.total_steps = 0
        self.best_steps_board = []
        
        self.actions_avg_list = []
        self.actions_deque = {
            0:deque(maxlen=50),
            1:deque(maxlen=50),
            2:deque(maxlen=50),
            3:deque(maxlen=50)
        }


        # Replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Initialize time step
        self.t_step = 0
        self.steps_ahead = predict_steps

    def non_monotonic_penalty(self, state):
        """Penalize the board state for having non-monotonic rows and columns."""
        penalty = 0
        for row in state:
            row_sorted = np.sort(row)
            row_penalty = min(np.sum(row != row_sorted), np.sum(row != row_sorted[::-1]))  # Count the number of tiles in the wrong order
            penalty += row_penalty

        for col in state.T:  # .T for transpose to get columns
            col_sorted = np.sort(col)
            col_penalty = min(np.sum(col != col_sorted), np.sum(col != col_sorted[::-1]))  # Count the number of tiles in the wrong order
            penalty += col_penalty

        return penalty * 0.1

    def calculate_merge_count_bonus(self, state):
        """Give a bonus for the number of potential merges on the board."""
        bonus = 0
        for row in state:
            for i in range(len(row) - 1):
                if row[i] == row[i+1] and row[i] != 0:
                    bonus += np.log2(row[i])*0.1

        for col in state.T:  # .T for transpose to get columns
            for i in range(len(col) - 1):
                if col[i] == col[i+1] and col[i] != 0:
                    bonus += np.log2(col[i])*0.1

        return bonus
    
    def calculate_edge_max_bonus(self, state):
        """ Grant bonus if the maximum number is on the edge """
        max_val = np.max(state)
        if max_val in state[0, :] or max_val in state[-1, :] or max_val in state[:, 0] or max_val in state[:, -1]:
            return self.edge_max_bonus
        else:
            return 0

    def calculate_open_square_bonus(self, state):
        """ Grant bonus for open squares """
        open_squares = np.sum(state == 0)
        return open_squares * self.open_square_bonus

    
    def evaluate(self, state):
        """Evaluate the state of the game board."""
        state = state.reshape(4,4)

        # The value of the state is the score minus the penalty plus the bonus
        value = self.calculate_edge_max_bonus(state) + self.calculate_open_square_bonus(state) - self.non_monotonic_penalty(state) + self.calculate_merge_count_bonus(state)

        return value

    def save(self, name):
        """Saves the state of the model and stats
        
        Params
        ======
            name (str): name of the agent version used in dqn function
        """
        
        torch.save(self.qnetwork_local.state_dict(), base_dir+'/network_local_%s.pth' % name)
        torch.save(self.qnetwork_target.state_dict(), base_dir+'/network_target_%s.pth' % name)
        torch.save(self.optimizer.state_dict(), base_dir+'/optimizer_%s.pth' % name)
        torch.save(self.lr_decay.state_dict(), base_dir+'/lr_schd_%s.pth' % name)
        state = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'losses': self.losses,
            'use_expected_rewards': self.use_expected_rewards,
            'current_iteration': self.current_iteration,
        
        # Game scores
            'scores_list': self.scores_list,
            'last_n_scores': self.last_n_scores,
            'mean_scores': self.mean_scores,
            'max_score': self.max_score,
            'min_score': self.min_score,
            'best_score_board': self.best_score_board,

        # Rewards
            'total_rewards_list': self.total_rewards_list,
            'last_n_total_rewards': self.last_n_total_rewards,
            'mean_total_rewards': self.mean_total_rewards,
            'max_total_reward': self.max_total_reward,
            'best_reward_board': self.best_reward_board,

        # Max cell value on game board
            'max_vals_list': self.max_vals_list,
            'last_n_vals': self.last_n_vals,
            'mean_vals': self.mean_vals,
            'max_val': self.max_val,
            'best_val_board': self.best_val_board,

        # Number of steps per episode
            'max_steps_list': self.max_steps_list,
            'last_n_steps': self.last_n_steps,
            'mean_steps': self.mean_steps,
            'max_steps': self.max_steps,
            'total_steps': self.total_steps,
            'best_steps_board': self.best_steps_board,
        
            'actions_avg_list': self.actions_avg_list,
            'actions_deque': self.actions_deque,
        # Replay buffer
            'memory': self.memory.dump(),
        # Initialize time step
            't_step': self.t_step,
            'steps_ahead': self.steps_ahead
        }

        with open(base_dir+'/agent_state_%s.pkl' % name, 'wb') as f:
            pickle.dump(state, f)

    def load(self, name):
        """Saves the state of the model and stats
        
        Params
        ======
            name (str): name of the agent version used in dqn function
        """
        self.qnetwork_local.load_state_dict(torch.load(base_dir+'/network_local_%s.pth' % name))
        self.qnetwork_target.load_state_dict(torch.load(base_dir+'/network_target_%s.pth' % name))
        self.optimizer.load_state_dict(torch.load(base_dir + '/optimizer_%s.pth' % name))
        self.lr_decay.load_state_dict(torch.load(base_dir + '/lr_schd_%s.pth' % name))
        
        with open(base_dir+'/agent_state_%s.pkl' % name, 'rb') as f:
            state = pickle.load(f)

        self.state_size = state['state_size']
        self.action_size = state['action_size']
        self.seed = state['seed']
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.batch_size = state['batch_size']
        self.losses = state['losses']
        self.use_expected_rewards = state['use_expected_rewards']
        self.current_iteration = state['current_iteration']
        
        # Game scores
        self.scores_list = state['scores_list']
        self.last_n_scores = state['last_n_scores']
        self.mean_scores = state['mean_scores']
        self.max_score = state['max_score']
        self.min_score = state['min_score'] if 'min_score' in state.keys() else state['max_score']
        self.best_score_board = state['best_score_board']

        # Rewards
        self.total_rewards_list = state['total_rewards_list']
        self.last_n_total_rewards = state['last_n_total_rewards']
        self.mean_total_rewards = state['mean_total_rewards']
        self.max_total_reward = state['max_total_reward']
        self.best_reward_board = state['best_reward_board']

        # Max cell value on game board
        self.max_vals_list = state['max_vals_list']
        self.last_n_vals = state['last_n_vals']
        self.mean_vals = state['mean_vals']
        self.max_val = state['max_val']
        self.best_val_board = state['best_val_board']

        # Number of steps per episode
        self.max_steps_list = state['max_steps_list']
        self.last_n_steps = state['last_n_steps']
        self.mean_steps = state['mean_steps']
        self.max_steps = state['max_steps']
        self.total_steps = state['total_steps']
        self.best_steps_board = state['best_steps_board']
        
        self.actions_avg_list = state['actions_avg_list']
        self.actions_deque = state['actions_deque']

        # Replay buffer
        self.memory.load(state['memory'])
        
        # Initialize time step
        self.t_step = state['t_step']
        self.steps_ahead = state['steps_ahead']

        
    def step(self, state, action, reward, next_state, done, error, action_dist):
        # Save experience in replay memory    
        self.memory.add(state, action, reward, next_state, done, error, action_dist, None)

    def get_children(self, state):
        """Generates all possible children states from a given state.

        Params
        ======
            state (array_like): current state
        """
        children = []
        for action in range(self.action_size):
            child = self.perform_action(state, action)
            if child is not None:
                children.append(child)
        return children
    
    def is_terminal(self, state):
        """Checks if a state is terminal.

        Params
        ======
            state (array_like): current state
        """
        if len(np.argwhere(state == 0)) > 0:
            # there are still empty positions, so it's not a terminal state
            return False

        for action in range(self.action_size):
            child = self.perform_action(state, action)
            if child is not None:
                # there is at least one valid action, so it's not a terminal state
                return False

        # no empty positions and no valid actions, so it's a terminal state
        return True
    
    def get_children_with_probabilities(self, state):
        """Generates all possible children states from a given state, along with their probabilities.

        Params
        ======
            state (array_like): current state
        """
        children = []
        empty_positions = np.argwhere(state == 0)
        new_values = [(2, 0.9), (4, 0.1)]  # new tile can be 2 with probability 0.9 or 4 with probability 0.1

        for action in range(self.action_size):
            child = self.perform_action(state, action)
            if child is not None:
                for position in empty_positions:
                    for value, probability in new_values:
                        new_child = child.copy()
                        new_child[position[0]][position[1]] = value
                        children.append((new_child, probability))  # store the child along with its probability

        return children
    
    def is_max_node(self, state):
        """Checks if a node is a max node.

        Params
        ======
            state (array_like): current state
        """
        # Assuming we're alternating between agent's turn and chance node (the game's turn to spawn new tiles).
        # If there are no empty positions, it's the agent's turn to make a move, so it's a max node.
        return len(np.argwhere(state == 0)) == 0



    def expectimax(self, state, depth):
        """Implementation of the expectimax algorithm for decision making.

        Params
        ======
            state (array_like): current state
            depth (int): depth of the expectimax tree
        """
        if depth == 0 or self.is_terminal(state):
            # print("evaluate:   ", self.evaluate(state)) # for debugging
            return self.evaluate(state)

        if self.is_max_node(state):
            max_value = float("-inf")
            for child in self.get_children(state):
                max_value = max(max_value, self.expectimax(child, depth - 1))
            # print("max_value:  ", max_value) # for debugging
            return max_value
        else:
            expected_value = 0
            for child, probability in self.get_children_with_probabilities(state):
                expected_value += probability * self.expectimax(child, depth - 1)
            # print("exp_value:  ", expected_value) # for debugging
            # print("depth:      ", depth) # for debugging    
            return expected_value
        
    def get_child(self, state, action):
        """Generates the child state from a given state and action.

        Params
        ======
            state (array_like): current state
            action (int): action to be performed on the state
        """
        return self.perform_action(state, action)

    def get_possible_actions(self, state):
        """Generates all possible actions from a given state.

        Params
        ======
            state (array_like): current state
        """
        possible_actions = []

        # For each action, check if it's a valid move
        for action in range(self.action_size):
            next_state = self.perform_action(state, action)
            # Check if the action has any effect on the state. 
            # If it doesn't, it's not a valid action and should not be included.
            if not np.array_equal(state, next_state):
                possible_actions.append(action)

        return possible_actions
    
    def slide(self, row):
        """
        Function to slide a row to the left, merging identical tiles.
        
        Params
        ======
            row (array_like): input row to slide
        """
        # Shift entries of each row to the left
        non_zero_elements = row[row != 0]  # remove zeros
        empty_elements = len(row) - len(non_zero_elements)
        new_row = np.zeros_like(row)
        new_row[:len(non_zero_elements)] = non_zero_elements
        
        # Merge entries and double their value and slide again
        for i in range(3):
            if new_row[i] == new_row[i + 1] != 0: 
                new_row[i] = 2 * new_row[i]
                new_row[i + 1:] = np.append(new_row[i + 2:], 0)
                
        return new_row
    
    def perform_action(self, state, action):
        """
        Perform a specific action on the current state and return the new state.
        
        Params
        ======
            state (array_like): current state
            action (int): action to be performed on the state
        """
        new_state = np.copy(state) # copy to not change original state
        
        # Map actions to directions
        directions = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        direction = directions[action]
        
        if direction == 'up':
            for i in range(4):
                new_state[:,i] = self.slide(new_state[:,i])
        elif direction == 'down':
            for i in range(4):
                new_state[:,i] = self.slide(np.flip(new_state[:,i]))[::-1]
        elif direction == 'left':
            for i in range(4):
                new_state[i,:] = self.slide(new_state[i,:])
        elif direction == 'right':
            for i in range(4):
                new_state[i,:] = self.slide(np.flip(new_state[i,:]))[::-1]
        
        return new_state



    def act(self, state):
        """Returns actions for given state as per expectimax algorithm.

        Params
        ======
            state (array_like): current state
        """
        actions = np.zeros(4)
        actions = np.array(actions)
        max_value = float("-inf")
        best_action = None
        state = state.reshape(4,4)
        # print(state)
        for action in self.get_possible_actions(state):
            child = self.get_child(state, action)
            value = self.expectimax(child, self.depth)
            actions[action] += value   
            if value > max_value:
                max_value = value
                best_action = action
        actions = actions[np.newaxis, :]
        return actions

                
    def soft_update(self, local_model, target_model, tau):
        """NOT USED ANYMORE
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        
        self.episode_memory = []
        self.batch_size = batch_size
        
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "error", "action_dist", "weight"])
        self.geomspaces = [np.geomspace(1., 0.5, i) for i in range(1, 10)]

    def dump(self):
        # Saves the buffer into dict object
        d = {
            'action_size': self.action_size,
             'batch_size': self.batch_size,
             'seed': self.seed,
             'geomspaces': self.geomspaces
        }

        d['memory'] = [d._asdict() for d in self.memory]
        return d

    def load(self, d):
        # creates a new buffer from dict
        self.action_size = d['action_size']
        self.batch_size = d['batch_size']
        self.seed = d['seed']
        self.geomspaces = d['geomspaces']

        for e in d['memory']:
            self.memory.append(self.experience(**e))

    def reset_episode_memory(self):
        self.episode_memory = []
        
    def add(self, state, action, reward, next_state, done, error, action_dist, weight = None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, error, action_dist, weight)
        self.episode_memory.append(e)

    def add_episode_experiences(self):
        self.memory.extend(self.episode_memory)
        self.reset_episode_memory()
        
    def calc_expected_rewards(self, steps_ahead = 1, weight = None):

        rewards = [e.reward for e in self.episode_memory if e is not None]

        exp_rewards = [np.sum(rewards[i:i+steps_ahead] * self.geomspaces[steps_ahead-1]) for i in range(len(rewards) - steps_ahead)]

        temp_memory = []
        
        for i, e in enumerate(self.episode_memory[:-steps_ahead]):
            t_e = self.experience(e.state, e.action, exp_rewards[i], e.next_state, e.done, e.error, e.action_dist, weight)
            temp_memory.append(t_e)

        self.episode_memory = temp_memory
            
    def sample(self, mode='board_max'):
        """Randomly sample a batch of experiences from memory."""
        
        if mode == 'random':
            experiences = random.sample(self.memory, k=self.batch_size)
        elif mode == 'board_max':
            probs = np.array([e.state.max() for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
                
        elif mode == 'board_sum':
            probs = np.array([e.state.sum() for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
                
        elif mode == 'reward':
            # Shifting by +1 is to keep steps with 0 reward in training set, otherwise they will receive 0 probability during sampling
            probs = np.array([e.reward + 1 for e in self.memory]) 
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
                
        elif mode == 'error':
            probs = np.array([e.error for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'error_u':
            probs = np.array([e.error for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'weighted_error':
            weights = np.array([e.weight for e in self.memory])
            max_weight = weights.max()
            sum_weight = weights.sum()
            weights = np.array([(max_weight - w + 1) / (sum_weight + len(weights)) for w in weights])
            probs = np.array([e.error for e in self.memory])
            probs = probs * weights
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'weighted_error_reversed':
            weights = np.array([e.weight for e in self.memory])
            sum_weight = weights.sum()
            weights = np.array([(w) / (sum_weight) for w in weights])
            probs = np.array([e.error for e in self.memory])
            probs = probs * weights
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'action_balanced_error':
            probs = np.array([e.error * e.action_dist for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'clipped_error':
            t = pd.DataFrame(self.memory)
            
            t = t[t['error'] < t['error'].quantile(0.99)]
            t['probs'] = t['error'] * t['action_dist']
            t['probs'] = t['probs'] / t['probs'].sum()
            idx = np.random.choice(len(t), size=self.batch_size, p=t['probs'].values)
            t = t.iloc[idx]
            
            experiences = deque(maxlen=self.batch_size)        
            for i in list(t.itertuples(name='Experience', index=False)):
                experiences.append(i)
        
     

        states = torch.from_numpy(np.vstack([transform_state(e.state, mode='conv') for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([transform_state(e.next_state, mode='conv') for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        # for state in states:
        #     print(state)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)