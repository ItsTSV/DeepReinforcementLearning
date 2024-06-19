import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Monitoring:
    def __init__(self, save_dir, save_name="Game-episode", mean_reward_interval=50):
        self.reward_history = []
        self.step_history = []
        self.df = pd.DataFrame(columns=['episode', 'reward', 'steps'])
        self.mean_reward_interval = mean_reward_interval
        self.save_name = save_name
        self.save_dir = save_dir

    def add_data(self, episode, reward, steps):
        self.reward_history.append(reward)
        self.step_history.append(steps)
        self.df.loc[len(self.df)] = [episode, reward, steps]

        if episode % self.mean_reward_interval == 0:
            self.df.to_csv(f'Data/{self.save_dir}/{self.save_name}{episode}.csv', index=False)

    def plot_rewards(self):
        plt.figure(3)
        plt.clf()
        plt.title('Rewards per episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(np.array(self.reward_history))
        plt.pause(0.001)

    def plot_steps(self):
        plt.figure(2)
        plt.clf()
        plt.title('Steps per episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.plot(np.array(self.step_history))
        plt.pause(0.001)
