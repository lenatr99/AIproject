import tkinter
from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c
import q_learning_agent as qla
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

def gen():
    return random.randint(0, c.GRID_LEN - 1)

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
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.update_grid_cells()
        self.scores = []

        self.ai_mode = ai_mode
        if self.ai_mode:
            self.agent = qla.QLearningAgent()
            self.train_agent()
            self.plot_scores()
        else:
            self.mainloop()

    def train_agent(self):
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()  # Create a figure and axis for the plot
        start_times = []
        end_times = []

        num_episodes = 100000
        for episode in range(num_episodes):  # Train for 10000 episodes
            start_time = datetime.datetime.now()
            start_times.append(start_time)
            self.matrix = logic.new_game(c.GRID_LEN)
            self.history_matrixs = []
            self.update_grid_cells()

            while True:
                state = self.matrix.copy()
                action = self.agent.choose_action(state, c.ACTIONS)
                action_done = self.perform_action(action)

                if action_done:
                    next_state = self.matrix.copy()
                    reward = logic.get_reward(state, action, next_state)
                    self.agent.learn(state, action, next_state, reward)
                    print(f"reward: {reward} epsilon: {self.agent.epsilon}\n")

                game_over = logic.game_state(self.matrix)[0] in ['win', 'lose']
                if game_over:
                    _, score = logic.game_state(self.matrix)
                    self.save_score(score)
                    end_time = datetime.datetime.now()
                    end_times.append(end_time)
                    self.plot_scores(fig, ax, start_times, end_times, episode)
                    break
            
            self.agent.decay_epsilon(num_episodes)

        plt.ioff()  # Disable interactive mode
        plt.show()  # Keep the plot window open

    def perform_action(self, action):
        if action in self.commands:
            self.matrix, done = self.commands[action](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
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

    def plot_scores(self, fig, ax, start_times, end_times, episode):
        ax.clear()  # Clear the axis
        x_values = list(range(episode + 1))  # X-axis represents the number of episodes
        color_values = [(end - start).total_seconds() / score if score != 0 else 0 for start, end, score in zip(start_times, end_times, self.scores)]  # Duration of each episode divided by the score
        scatter = ax.scatter(x_values, self.scores, c=color_values, cmap='viridis', marker='o')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Scores over Episodes')

        if episode == 0:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Duration/Score (seconds)")

        plt.draw()  # Redraw the plot
        plt.pause(0.001)  # Pause to allow the plot to update



game_grid = GameGrid(ai_mode=True)