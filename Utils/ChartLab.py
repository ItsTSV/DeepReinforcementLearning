import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
    This file is *NOT* a part of algorithm implementation. It is only used to draw charts and calculate data.
    that will be used in the 'Training' chapter of my thesis.
'''

'''
    *
    *   Loading and preprocessing section
    *
'''
# Load datasets
df_dqn_pong = pd.read_csv("../FinalData/DQN/Pong-episode-300.csv")
df_ddqn_pong = pd.read_csv("../FinalData/DDQN/Pong-episode-300.csv")
df_duelingpdqn_pong = pd.read_csv("../FinalData/DuelingPDQN/Pong-episode-300.csv")
df_dqn_boxing = pd.read_csv("../FinalData/DQN/Boxing-episode-800.csv")
df_ddqn_boxing = pd.read_csv("../FinalData/DDQN/Boxing-episode-800.csv")
df_duelingpdqn_boxing = pd.read_csv("../FinalData/DuelingPDQN/Boxing-episode-800.csv")

sum_dqn_pong = df_dqn_pong["steps"].sum()
sum_ddqn_pong = df_ddqn_pong["steps"].sum()
sum_duelingpdqn_pong = df_duelingpdqn_pong["steps"].sum()
print("Mean steps in pong: ", (sum_ddqn_pong + sum_dqn_pong + sum_duelingpdqn_pong) / 3)
print("Sum: ", sum_dqn_pong, sum_duelingpdqn_pong, sum_ddqn_pong)

sum_dqn_boxing = df_dqn_boxing["steps"].sum()
sum_ddqn_boxing = df_ddqn_boxing["steps"].sum()
sum_duelingpdqn_boxing = df_duelingpdqn_boxing["steps"].sum()
print("Mean steps in boxing: ", (sum_ddqn_boxing + sum_dqn_boxing + sum_duelingpdqn_boxing) / 3)
print("Sum: ", sum_dqn_boxing, sum_duelingpdqn_boxing, sum_ddqn_boxing)

# Reduce dimensions so the charts are readable
df_dqn_pong = df_dqn_pong[::5]
df_dqn_pong["new_reward"] = df_dqn_pong.groupby("episode")["reward"].transform("mean")
df_dqn_pong["new_steps "] = df_dqn_pong.groupby("episode")["steps"].transform("sum")
df_dqn_pong = df_dqn_pong.drop(columns=["reward", "steps"])

df_ddqn_pong = df_ddqn_pong[::5]
df_ddqn_pong["new_reward"] = df_ddqn_pong.groupby("episode")["reward"].transform("mean")
df_ddqn_pong["new_steps "] = df_ddqn_pong.groupby("episode")["steps"].transform("sum")
df_ddqn_pong = df_ddqn_pong.drop(columns=["reward", "steps"])

df_duelingpdqn_pong = df_duelingpdqn_pong[::5]
df_duelingpdqn_pong["new_reward"] = df_duelingpdqn_pong.groupby("episode")["reward"].transform("mean")
df_duelingpdqn_pong["new_steps "] = df_duelingpdqn_pong.groupby("episode")["steps"].transform("sum")
df_duelingpdqn_pong = df_duelingpdqn_pong.drop(columns=["reward", "steps"])

df_dqn_boxing = df_dqn_boxing[::10]
df_dqn_boxing["new_reward"] = df_dqn_boxing.groupby("episode")["reward"].transform("mean")
df_dqn_boxing["new_steps "] = df_dqn_boxing.groupby("episode")["steps"].transform("sum")
df_dqn_boxing = df_dqn_boxing.drop(columns=["reward", "steps"])

df_ddqn_boxing = df_ddqn_boxing[::10]
df_ddqn_boxing["new_reward"] = df_ddqn_boxing.groupby("episode")["reward"].transform("mean")
df_ddqn_boxing["new_steps "] = df_ddqn_boxing.groupby("episode")["steps"].transform("sum")
df_ddqn_boxing = df_ddqn_boxing.drop(columns=["reward", "steps"])

df_duelingpdqn_boxing = df_duelingpdqn_boxing[::10]
df_duelingpdqn_boxing["new_reward"] = df_duelingpdqn_boxing.groupby("episode")["reward"].transform("mean")
df_duelingpdqn_boxing["new_steps "] = df_duelingpdqn_boxing.groupby("episode")["steps"].transform("sum")
df_duelingpdqn_boxing = df_duelingpdqn_boxing.drop(columns=["reward", "steps"])

'''
    *
    * Training section
    *
'''
# Draw charts for Pong using Seaborn
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 5))
plt.xlabel("Episode")
plt.ylabel("Episode reward")
# Add lines for each agent
sns.lineplot(data=df_dqn_pong, x="episode", y="new_reward", label="DQN", alpha=0.8, linewidth=2)
sns.lineplot(data=df_ddqn_pong, x="episode", y="new_reward", label="DDQN", alpha=0.8, linewidth=2)
sns.lineplot(data=df_duelingpdqn_pong, x="episode", y="new_reward", label="DuelingPDQN", alpha=0.8, linewidth=2)
# Add lines for mean rewards
plt.axhline(y=np.mean(df_dqn_pong["new_reward"]), color='blue', linestyle='--', label="DQN mean reward", linewidth=2,
            alpha=0.8)
plt.axhline(y=np.mean(df_ddqn_pong["new_reward"]), color='orange', linestyle='--', label="DDQN mean reward",
            linewidth=2, alpha=0.8)
plt.axhline(y=np.mean(df_duelingpdqn_pong["new_reward"]), color='green', linestyle='--',
            label="DuelingPDQN mean reward", linewidth=2, alpha=0.8)
plt.savefig("../FinalData/Charts/pongTraining.png")

# Draw charts for Boxing using Seaborn
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 5))
plt.xlabel("Episode")
plt.ylabel("Episode reward")
# Add lines for each agent
sns.lineplot(data=df_dqn_boxing, x="episode", y="new_reward", label="DQN", alpha=0.8, linewidth=2)
sns.lineplot(data=df_ddqn_boxing, x="episode", y="new_reward", label="DDQN", alpha=0.8, linewidth=2)
sns.lineplot(data=df_duelingpdqn_boxing, x="episode", y="new_reward", label="DuelingPDQN", alpha=0.8, linewidth=2)
# Add lines for mean rewards
plt.axhline(y=np.mean(df_dqn_boxing["new_reward"]), color='blue', linestyle='--', label="DQN mean reward", linewidth=2,
            alpha=0.8)
plt.axhline(y=np.mean(df_ddqn_boxing["new_reward"]), color='orange', linestyle='--', label="DDQN mean reward",
            linewidth=2, alpha=0.8)
plt.axhline(y=np.mean(df_duelingpdqn_boxing["new_reward"]), color='green', linestyle='--',
            label="DuelingPDQN mean reward", linewidth=2, alpha=0.8)
plt.savefig("../FinalData/Charts/boxingTraining.png")

'''
    *
    *   Testing section
    *
'''
# Load datasets
df_dqn_testing_pong = pd.read_csv("../FinalData/TestLogs/DQN-Pong-v5-testing.csv")
df_ddqn_testing_pong = pd.read_csv("../FinalData/TestLogs/DDQN-Pong-v5-testing.csv")
df_duelingpdqn_testing_pong = pd.read_csv("../FinalData/TestLogs/DuelingPDQN-Pong-v5-testing.csv")
df_random_testing_pong = pd.read_csv("../FinalData/TestLogs/Random-Pong-v5-testing.csv")
df_dqn_testing_boxing = pd.read_csv("../FinalData/TestLogs/DQN-Boxing-v5-testing.csv")
df_ddqn_testing_boxing = pd.read_csv("../FinalData/TestLogs/DDQN-Boxing-v5-testing.csv")
df_duelingpdqn_testing_boxing = pd.read_csv("../FinalData/TestLogs/DuelingPDQN-Boxing-v5-testing.csv")
df_random_testing_boxing = pd.read_csv("../FinalData/TestLogs/Random-Boxing-v5-testing.csv")

# Draw charts for Pong testing using Seaborn
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 5))
plt.xlabel("Episode steps")
plt.ylabel("Episode reward")
# Scatter plot for each agent
sns.scatterplot(data=df_dqn_testing_pong, x="steps", y="reward", label="DQN", alpha=0.7, s=100)
sns.scatterplot(data=df_ddqn_testing_pong, x="steps", y="reward", label="DDQN", alpha=0.7, s=100)
sns.scatterplot(data=df_duelingpdqn_testing_pong, x="steps", y="reward", label="DuelingPDQN", alpha=0.7, s=100)
sns.scatterplot(data=df_random_testing_pong, x="steps", y="reward", label="Random", alpha=0.7, s=100)
plt.savefig("../FinalData/Charts/pongTesting.png")

# Draw charts for Boxing testing using Seaborn
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 5))
plt.xlabel("Episode steps")
plt.ylabel("Episode reward")
# Scatter plot for each agent
sns.scatterplot(data=df_dqn_testing_boxing, x="steps", y="reward", label="DQN", alpha=0.7, s=100)
sns.scatterplot(data=df_ddqn_testing_boxing, x="steps", y="reward", label="DDQN", alpha=0.7, s=100)
sns.scatterplot(data=df_duelingpdqn_testing_boxing, x="steps", y="reward", label="DuelingPDQN", alpha=0.7, s=100)
sns.scatterplot(data=df_random_testing_boxing, x="steps", y="reward", label="Random", alpha=0.7, s=100)
plt.savefig("../FinalData/Charts/boxingTesting.png")
