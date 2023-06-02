import numpy as np
import random
import math

base_dir = './data/'

PERFECT_SNAKE = [[2,   2**2, 2**3, 2**4],
                [2**8, 2**7, 2**6, 2**5],
                [2**9, 2**10,2**11,2**12],
                [2**16,2**15,2**14,2**13]]

class Agent():
    """ Interacts with and learns from the environment """

    def __init__(self, state_size = 4*4, action_size = 4, seed = 42,
                 edge_max_bonus=0.5, open_square_bonus=0.2, corner_max_penalty_value=0.0,
                 non_monotonic_penalty_value=0.1, blocking_penalty_value=0.1):
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
        self.edge_max_bonus = edge_max_bonus  # Bonus for having large values on the edge
        self.open_square_bonus = open_square_bonus  # Bonus for open squares
        self.corner_max_penalty_value = corner_max_penalty_value  # Penalty for not having tha max tile in the corner
        self.non_monotonic_penalty_value = non_monotonic_penalty_value  # Penalty for non-monotonic rows and columns
        self.blocking_penalty_value = blocking_penalty_value  # Penalty for blocking high valued tiles
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.total_steps = 0
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

        return penalty * self.non_monotonic_penalty_value

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
    
    def corner_max_penalty(self, state):
        """ Penalize the board state for having the max value not at a corner """
        penalty = 0
        max_val = np.max(state)
        sorted_values = np.unique(np.sort(state))  # Get the sorted unique values in the state
        # on which index is the max value
        max_index = np.where(sorted_values == max_val)[0][0]
              
                
        # Check if max value is at any of the four corners
        if not (state[0,0] == max_val or state[0,3] == max_val or state[3,0] == max_val or state[3,3] == max_val):
            penalty = self.corner_max_penalty_value  # Arbitrary penalty value
        
        return penalty * max_index
    
    def blocking_penalty(self, state):
        if self.blocking_penalty_value == 0:
            return 0
        
        """Calculate a penalty if high valued tiles are blocked by low valued tiles."""
        penalty = 0
        highest_tile = np.max(state)

        if highest_tile < 200: # to avoid penalty for low valued tiles
            return 0
        
        # 2D array to get all directions around a cell for up, down, left and right
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        # scan the board
        for i in range(self.action_size):
            for j in range(self.action_size):
                if state[i][j] == highest_tile:
                    # Check all directions
                    for direction in directions:
                        ni, nj = i + direction[0], j + direction[1]
                        # Check if neighbour indices are in the board range
                        if ni >= 0 and ni < self.action_size and nj >= 0 and nj < self.action_size:
                            # Check if neighbouring tile has a lower value
                            if state[ni][nj] < state[i][j]:
                                # Add to penalty - The higher the difference, the higher the penalty
                                penalty += (state[i][j] - state[ni][nj]) * self.blocking_penalty_value

        return penalty
    def reward_adjacent_x2_tiles(self, state):
        """Rewards situations where a tile is adjacent to another tile of double its value"""
        if self.reward_adjacent_x2_tiles == 0:
            return 0
        
        highest_tile = np.max(state)

        if highest_tile < 128: # to avoid penalty for low valued tiles
            return 0

        reward = 0
        # 2D array to get all directions around a cell for up, down, left and right
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        # scan the board
        for i in range(self.action_size):
            for j in range(self.action_size):
                if state[i][j] >= 128:
                    # Check all directions
                    for direction in directions:
                        ni, nj = i + direction[0], j + direction[1]
                        # Check if neighbour indices are in the board range
                        if ni >= 0 and ni < self.action_size and nj >= 0 and nj < self.action_size:
                            # Check if neighbouring tile is double
                            if state[ni][nj] == 2 * state[i][j] or state[i][j] == 2 * state[ni][nj]:
                                # Add to reward - the higher the tile, the higher the reward
                                reward += max(state[ni][nj], state[i][j]) * self.reward_adjacent_x2_tiles_value
        return reward

    
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
    
    
    def snakeHeuristic(self, state):
        h = 0
        for i in range(self.action_size):
            for j in range(self.action_size):
                h += state[i][j] * PERFECT_SNAKE[i][j]

        return h
    
    def expectiminimax(self, board, depth, direction = None):
        if board.check_loss():  
            return -math.inf, direction
        elif depth < 0:
            return self.snakeHeuristic(board), direction

        a = 0
        if depth != int(depth):
            # Player's turn, pick max
            a = -math.inf
            for direction in range(self.action_size):
                new_board, _, done, _ = board.step(direction)
                if not done:
                    result = self.expectiminimax(new_board, depth - 0.5, direction)[0]
                    if result > a:
                        a = result
        elif depth == int(depth):
            # Nature's turn, calculate average
            a = 0
            open_positions = board.get_open_positions()
            for pos in open_positions:
                board.add_tile(pos, 2)
                a += 1.0 / len(open_positions) * self.expectiminimax(board, depth - 0.5, direction)[0]
                board.add_tile(pos, 0)  # Reset the position to be free
        return (a, direction)

    
    def evaluate(self, state):
        """Evaluate the state of the game board."""
        state = state.reshape(4,4)

        value = self.snakeHeuristic(state)

        """# The value of the state is the score minus the penalty plus the bonus
        value = self.calculate_edge_max_bonus(state) 
        + self.calculate_open_square_bonus(state) 
        - self.non_monotonic_penalty(state) 
        + self.calculate_merge_count_bonus(state) 
        #- self.corner_max_penalty(state) 
        - self.blocking_penalty(state) # Included corner max penalty
        + self.reward_adjacent_x2_tiles(state)

        debug = True
        if np.max(state) >= 128 and debug:
            print("State: \n", state)
            print("Edge max bonus: ", self.calculate_edge_max_bonus(state))
            print("Open square bonus: ", self.calculate_open_square_bonus(state))
            print("Non monotonic penalty: ", self.non_monotonic_penalty(state))
            print("Merge count bonus: ", self.calculate_merge_count_bonus(state))
            print("Corner max penalty: ", self.corner_max_penalty(state))
            print("Blocking penalty: ", self.blocking_penalty(state))
            print("Reward adjacent x2 tiles: ", self.reward_adjacent_x2_tiles(state))
            print("Value: ", value)
"""
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



    def expectimax(self, state, depth, alpha=float('-inf'), beta=float('inf')):
        """Implementation of the expectimax algorithm for decision making.

        Params
        ======
            state (array_like): current state
            depth (int): depth of the expectimax tree
        """
        if depth == 0 or self.is_terminal(state):
            # print("evaluate:   ", self.evaluate(state)) # for debugging
            return self.evaluate(state)

        # ADDED ALPHA-BETA PRUNING
        if self.is_max_node(state):
            value = float("-inf")
            for child in self.get_children(state):
                value = max(value, self.expectimax(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # Beta cut-off
            return value
        
        else:
            value = 0
            for child, probability in self.get_children_with_probabilities(state):
                value += probability * self.expectimax(child, depth - 1, alpha, beta)
                if value >= beta: 
                    return value  # Alpha cut-off
                alpha = max(alpha, value)
            return value
            
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



    def act(self, state, depth=2):
        """Returns actions for given state as per expectimax algorithm.

        Params
        ======
            state (array_like): current state
        """
        """actions = np.zeros(4)
        actions = np.array(actions)
        max_value = float("-inf")
        state = state.reshape(4,4)
        alpha = float('-inf')
        beta = float('inf')
        for action in self.get_possible_actions(state):
            child = self.get_child(state, action)
            value = self.expectimax(child, self.depth, alpha, beta)
            actions[action] += value 
            if value > max_value:
                max_value = value
                alpha = max_value
        actions = actions[np.newaxis, :]"""

        return self.expectiminimax(state, depth)[1] 
