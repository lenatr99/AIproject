import numpy as np
import random
import constants as c
import math

#######
# Task 1a #
#######

def new_game(n):
    matrix = np.zeros((n,n))
    matrix = add_two(matrix)
    matrix = add_two(matrix)
    return matrix

###########
# Task 1b #
###########

def add_two(mat):
    a = random.randint(0, len(mat)-1)
    b = random.randint(0, len(mat)-1)
    while mat[a][b] != 0:
        a = random.randint(0, len(mat)-1)
        b = random.randint(0, len(mat)-1)
    mat[a][b] = 2
    return mat

###########
# Task 1c #
###########

def game_state(mat):
    score = np.sum(mat)
    # check for win cell
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 2048:
                return 'win', score
    # check for any zero entries
    if np.any(mat == 0):
        return 'not over', score
    # check for same cells that touch each other
    if np.any(mat[:-1,:] == mat[1:,:]) or np.any(mat[:,:-1] == mat[:,1:]):
        return 'not over', score
    return 'lose', score

###########
# Task 2a #
###########

def reverse(mat):
    return np.flip(mat, axis=1)

###########
# Task 2b #
###########

def transpose(mat):
    return mat.T

##########
# Task 3 #
##########

def cover_up(mat):
    new = np.zeros((c.GRID_LEN, c.GRID_LEN))
    done = False
    for i in range(c.GRID_LEN):
        count = 0
        for j in range(c.GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done

def merge(mat, done):
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0
                done = True
    return mat, done

def up(game):
    game = transpose(game)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(game)
    return game, done

def down(game):
    game = np.flip(transpose(game), axis=0)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(np.flip(game, axis=0))
    return game, done

def left(game):
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    return game, done

def right(game):
    game = np.flip(game, axis=1)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = np.flip(game, axis=1)
    return game, done

def findemptyCell(mat):
    return np.count_nonzero(mat==0)

def get_reward(state, action, next_state):
    max_tile = np.max(state)
    next_max_tile = np.max(next_state)
    reward = 0

    #get number of merges
    empty1 = findemptyCell(state)
    empty2 = findemptyCell(next_state)

    #reward math.log(next_max,2)*0.1 if next_max is higher than prev max
    if next_max_tile > max_tile:
        reward += math.log(next_max_tile,2)*0.1

    #reward for number of merges
    reward += (empty2-empty1)

    return reward
