import tkinter
import tensorflow as tf
from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c
import dqn_agent as dqa  
import time
import pickle
import matplotlib.pyplot as plt
import datetime
import numpy as np
import threading
import queue
import matplotlib
import math
from keras.utils import to_categorical
matplotlib.use('Agg')


def gen():
    return random.randint(0, c.GRID_LEN - 1)

def change_values(X):
    power_mat = np.zeros(shape=(1,4,4,16),dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(X[i][j]==0):
                power_mat[0][i][j][0] = 1.0
            else:
                power = int(math.log(X[i][j],2))
                power_mat[0][i][j][power] = 1.0
    return power_mat  

def one_hot_encode(state, num_states):
    # Flatten the grid
    flattened = state.flatten()

    # One-hot encode each cell in the flattened grid
    one_hot_encoded = to_categorical(flattened, num_classes=num_states)

    return one_hot_encoded

class GameGrid(Frame):
    def __init__(self, ai_mode=False):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
        }

        self.grid_cells = []
        # self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        # self.update_grid_cells()
        self.scores = []


        self.ai_mode = ai_mode
        if self.ai_mode:
            self.agent = dqa.DQNAgent(c.GRID_LEN * c.GRID_LEN, len(self.commands))  # Initialize DQN Agent
            self.train_agent()
            
        else:
            self.mainloop()

    # Rest of the code remains the same..


    def train_agent(self):
        fig, ax = plt.subplots()  # Create a figure and axis for the plot
        total_iters = 1
        num_episodes = 20000
        for episode in range(num_episodes):  # Train for 1000 episodes
            self.matrix = logic.new_game(c.GRID_LEN)
            self.history_matrixs = []
            # self.update_grid_cells()

            while True:
                state = self.matrix.copy()
                print(one_hot_encode(state, 16))
                state_1 = change_values(state)
                print(state_1.shape)
                # state_1 = np.array(state,dtype = np.float32).reshape(1,4,4,16)
                action = self.agent.choose_action(state_1)
                action_2 = c.ACTIONS[action]
                action_done = self.perform_action(action_2)
                print(state)

                if action_done:
                    next_state = self.matrix.copy()
                    next_state_1 = change_values(next_state) # flatten the state
                    reward = logic.get_reward(state, action, next_state)
                    game_over = logic.game_state(self.matrix)[0] in ['win', 'lose']
                    self.agent.learn(state_1, action, reward, next_state_1, game_over, episode, total_iters)
                    print(f"iteration {episode} reward: {reward} epsilon: {self.agent.epsilon} total iters: {total_iters}\n")

                game_over = logic.game_state(self.matrix)[0] in ['win', 'lose']
                if game_over:
                    _, score = logic.game_state(self.matrix)
                    self.save_score(score)
                    # self.plot_scores(fig, ax, episode)
                    break
                total_iters += 1

    def perform_action(self, action):
        if action in self.commands:
            self.matrix, done = self.commands[action](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                self.history_matrixs.append(self.matrix)
                # self.update_grid_cells()
            return done
        return False

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="",bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT: exit()
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                if logic.game_state(self.matrix)[0] == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix)[0] == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def generate_next(self):
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2

    def save_score(self, score):
        self.scores.append(score)
        with open('scores.pickle', 'wb') as f:
            pickle.dump(self.scores, f)

    def plot_scores(self, fig, ax, episode):
        ax.clear()  # Clear the axis
        x_values = list(range(episode + 1))  # X-axis represents the number of episodes
        ax.plot(x_values, self.scores)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Scores over Episodes')
        plt.savefig('plot.png')

        # plt.draw()  # Redraw the plot
        plt.pause(0.001)  # Pause to allow the plot to update

game_grid = GameGrid(ai_mode=True)