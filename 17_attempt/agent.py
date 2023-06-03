import numpy as np
import random

base_dir = './data/'


class Agent():
    """ Interacts with and learns from the environment """

    def __init__(self, state_size = 4*4, action_size = 4, seed = 42,
                 corner_max_bonus=1.5, edge_max_bonus=0.5, open_square_bonus=0.2):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            edge_max_bonus (float): bonus for having large values on the edge
            open_square_bonus (float): bonus for open squares
            
        """
        self.depth = 2
        self.corner_max_bonus = corner_max_bonus
        self.edge_max_bonus = edge_max_bonus  # Bonus for having large values on the edge
        self.open_square_bonus = open_square_bonus  # Bonus for open squares
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)


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
        return -penalty * 0.5

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
        if max_val == state[0, 0] or max_val == state[-1, 0] or max_val == state[-1, -1] or max_val == state[0, -1]:
            return self.corner_max_bonus
        elif max_val in state[0, :] or max_val in state[-1, :] or max_val in state[:, 0] or max_val in state[:, -1]:
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
        # print(self.calculate_edge_max_bonus(state), self.calculate_open_square_bonus(state), self.non_monotonic_penalty(state), self.calculate_merge_count_bonus(state) )
        # The value of the state is the score minus the penalty plus the bonus
        value = self.calculate_edge_max_bonus(state) + self.calculate_open_square_bonus(state) + self.non_monotonic_penalty(state) + self.calculate_merge_count_bonus(state)

        return value

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
        state = state.reshape(4,4)
        # print(state)
        for action in self.get_possible_actions(state):
            child = self.get_child(state, action)
            value = self.expectimax(child, self.depth)
            actions[action] += value   
        actions = actions[np.newaxis, :]
        return actions
