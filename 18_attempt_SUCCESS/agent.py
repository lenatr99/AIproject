import numpy as np
import random
import time
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

base_dir = './data/'



action_size = 4 

seed = 65
random.seed(seed)
np.random.seed(seed)


def act(state, pool, depth):
    """Returns actions for given state as per expectimax algorithm.

    Params
    ======
        state (array_like): current state
    """
    actions = np.zeros(4)
    actions = np.array(actions)
    max_value = float("-inf")
    results = []
    state = state.reshape(4,4)
    # print(state)

    # For each possible action, apply it to get a child state
    child_states = [get_child(state, action) for action in get_possible_actions(state)]
    
    # Create a partial function to pass 'depth' to 'expectimax' function
    expectimax_func = partial(expectimax, depth=depth)

    # Use the pool to compute expectimax values of all children concurrently
    expectimax_values = pool.map(expectimax_func, child_states)
    
    # Update the actions array
    for i, action in enumerate(get_possible_actions(state)):
        actions[action] += expectimax_values[i]
    
    actions = actions[np.newaxis, :]
    # print(value)
    return actions


def evaluate(state):
    """Evaluate the state of the game board."""
    state = state.reshape(4, 4)
    value = 0.0

    # Edge weights

    edgeValue = np.array([[2,   2**2, 2**3, 2**4],
            [2**8, 2**7, 2**6, 2**5],
            [2**9, 2**10,2**11,2**12],
            [2**16,2**15,2**14,2**13]])
    
    for i in range(4):
        for j in range(4):
            if state[i][j] != 0:
                value += state[i][j] * edgeValue[i][j]

    return value


def get_children(state):
    """Generates all possible children states from a given state.

    Params
    ======
        state (array_like): current state
    """
    children = []
    for action in range(action_size):
        child = perform_action(state, action)
        if child is not None:
            children.append(child)
    return children

def is_terminal(state):
    """Checks if a state is terminal.

    Params
    ======
        state (array_like): current state
    """
    if len(np.argwhere(state == 0)) > 0:
        # there are still empty positions, so it's not a terminal state
        return False

    for action in range(action_size):
        child = perform_action(state, action)
        if child is not None:
            # there is at least one valid action, so it's not a terminal state
            return False

    # no empty positions and no valid actions, so it's a terminal state
    return True

def get_children_with_probabilities(state):
    """Generates all possible children states from a given state, along with their probabilities.

    Params
    ======
        state (array_like): current state
    """
    children = []
    empty_positions = np.argwhere(state == 0)
    new_values = [(2, 0.9), (4, 0.1)]  # new tile can be 2 with probability 0.9 or 4 with probability 0.1

    for action in range(action_size):
        child = perform_action(state, action)
        if child is not None:
            for position in empty_positions:
                new_child = child.copy()
                new_child[position[0]][position[1]] = 2
                children.append(new_child)  # store the child along with its probability

    return children

def is_max_node(state):
    """Checks if a node is a max node.

    Params
    ======
        state (array_like): current state
    """
    # Assuming we're alternating between agent's turn and chance node (the game's turn to spawn new tiles).
    # If there are no empty positions, it's the agent's turn to make a move, so it's a max node.
    return len(np.argwhere(state == 0)) == 0

    
def get_child(state, action):
    """Generates the child state from a given state and action.

    Params
    ======
        state (array_like): current state
        action (int): action to be performed on the state
    """
    return perform_action(state, action)

def get_possible_actions(state):
    """Generates all possible actions from a given state.

    Params
    ======
        state (array_like): current state
    """
    possible_actions = []

    # For each action, check if it's a valid move
    for action in range(action_size):
        next_state = perform_action(state, action)
        # Check if the action has any effect on the state. 
        # If it doesn't, it's not a valid action and should not be included.
        if not np.array_equal(state, next_state):
            possible_actions.append(action)

    return possible_actions

def slide(row):
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

def perform_action(state, action):
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
            new_state[:,i] = slide(new_state[:,i])
    elif direction == 'down':
        for i in range(4):
            new_state[:,i] = slide(np.flip(new_state[:,i]))[::-1]
    elif direction == 'left':
        for i in range(4):
            new_state[i,:] = slide(new_state[i,:])
    elif direction == 'right':
        for i in range(4):
            new_state[i,:] = slide(np.flip(new_state[i,:]))[::-1]
    
    return new_state




def expectimax(state, depth):
    """Implementation of the expectimax algorithm for decision making.

    Params
    ======
        state (array_like): current state
        depth (int): depth of the expectimax tree
    """
    print("depth: {}".format(depth))
    if depth == 0 or is_terminal(state):
        # print("evaluate:   ", evaluate(state)) # for debugging
        return evaluate(state)

    if is_max_node(state):
        max_value = float("-inf")
        for child in get_children(state):
            max_value = max(max_value, expectimax(child, depth - 1))
        return max_value
    else:
        expected_value = 0
        for child in get_children_with_probabilities(state):
            expected_value += expectimax(child, depth - 1)
        # print("exp_value:  ", expected_value) # for debugging
        # print("depth:      ", depth) # for debugging    
        return expected_value



