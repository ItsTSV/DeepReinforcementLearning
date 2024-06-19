import gymnasium as gym
import torch
import numpy as np
import argparse as ap
import pandas as pd
from DeepQLearning.DDQNAgent import DDQNAgent
from DeepQLearning.DQN import DQN
from DeepQLearning.DuelingDQN import DuelingDQN
from DeepQLearning.DuelingPDQNAgent import DuelingPDQNAgent
from DeepQLearning.DQNAgent import DQNAgent
from Printer import Printer

'''
    This file is *NOT* a part of algorithm implementation. It is only used to run, visualize the game, test agents
    and calculate their mean rewards that will be used in the 'Training' chapter of my thesis.
'''

# Set and parse arguments
parser = ap.ArgumentParser(description="Deep Q Learning, Double Deep Q Learning and Dueling Deep Q Learning with "
                                       "Prioritized Experience Replay implementation -- Test an agent."
                                       "All arguments are optional, default values are set.")
parser.add_argument("-algorithm", type=str, default="DQN", help="Algorithm to use. Can be DQN, DDQN or DuelingPDQN.")
parser.add_argument("-env_name", type=str, default="ALE/Pong-v5", help="Environment name.")
parser.add_argument("-network_path", type=str, default="../FinalData/DQN/Pong-episode-Max.pth")
parser.add_argument("-mode", type=str, default="show", help="Mode to run the game. Can be show or calculate.")
parser.add_argument("-episodes", type=int, default=50, help="Number of episodes to run the game.")
args = parser.parse_args()

# Validate arguments
if args.algorithm not in ["DQN", "DDQN", "DuelingPDQN"]:
    Printer.print_error("Algorithm must be DQN, DDQN or DuelingPDQN!")
    exit(1)
if args.env_name[:4] != "ALE/":
    Printer.print_error("Environment name must start with ALE/!")
    exit(1)
if args.mode not in ["show", "calculate", "random"]:
    Printer.print_error("Mode must be show, random or calculate!")
    exit(1)
if args.episodes <= 0:
    Printer.print_error("Number of episodes must be greater than 0!")
    exit(1)

# Print info about path
Printer.print_info(f"Network imported from path: {args.network_path}")

# Init device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    Printer.print_success("CUDA is available, using GPU.")
else:
    Printer.print_error(
        "CUDA is not available, using CPU! This will result in very slow training! Consider using a GPU.")

# Init environment with 0.1 chance to sticky action
render_like = "human" if args.mode == "show" else "rgb_array"
env = gym.make(args.env_name, render_mode=render_like, repeat_action_probability=0.1)
env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
env = gym.wrappers.FrameStack(env, 4)

# Environment variables
state: np.ndarray = env.reset()[0]
action_count: int = env.action_space.n
observation_count: int = len(state)

# Create agent based on algorithm
if args.algorithm == "DQN":
    policy_network = DQN(observation_count, action_count, device).to(device)
    target_network = DQN(observation_count, action_count, device).to(device)
    policy_network.load_state_dict(torch.load(args.network_path, map_location=torch.device(device)))
    target_network.load_state_dict(torch.load(args.network_path, map_location=torch.device(device)))
    agent = DQNAgent(policy_network, target_network, env, device)
elif args.algorithm == "DDQN":
    policy_network = DQN(observation_count, action_count, device).to(device)
    target_network = DQN(observation_count, action_count, device).to(device)
    policy_network.load_state_dict(torch.load(args.network_path, map_location=torch.device(device)))
    target_network.load_state_dict(torch.load(args.network_path, map_location=torch.device(device)))
    agent = DDQNAgent(policy_network, target_network, env, device)
elif args.algorithm == "DuelingPDQN":
    policy_network = DuelingDQN(observation_count, action_count, device).to(device)
    target_network = DuelingDQN(observation_count, action_count, device).to(device)
    policy_network.load_state_dict(torch.load(args.network_path, map_location=torch.device(device)))
    target_network.load_state_dict(torch.load(args.network_path, map_location=torch.device(device)))
    agent = DuelingPDQNAgent(policy_network, target_network, env, device)
else:
    Printer.print_error("Algorithm must be DQN, DDQN or DuelingPDQN!")
    exit(1)

# Run and visualize the game
episodes = 1 if args.mode == "show" else args.episodes
rewards = []
steps = []
for i in range(episodes):
    state = env.reset()[0]
    state = np.array(state, dtype=np.float32)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    episode_reward = 0
    episode_steps = 0

    while not done:
        env.render()
        if args.mode == "random":
            action = agent.random_action()
        else:
            action = agent.best_action(state)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        episode_steps += 1
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    rewards.append(episode_reward)
    steps.append(episode_steps)
    Printer.print_success(f"Episode {i} finished with score {episode_reward} after {episode_steps} steps.")

Printer.print_info("\n----------------------------------------")
Printer.print_info(f"Mean reward over {episodes} episodes: {np.mean(rewards)}")
Printer.print_info(f"Median reward over {episodes} episodes: {np.median(rewards)}")
Printer.print_info(f"Max reward over {episodes} episodes: {np.max(rewards)}")
Printer.print_info(f"Min reward over {episodes} episodes: {np.min(rewards)}")
Printer.print_info("----------------------------------------")

# Save rewards to csv (for further plotting)
if args.mode == "calculate" or args.mode == "random":
    to_df = zip(rewards, steps)
    df = pd.DataFrame(to_df, columns=["reward", "steps"])
    algorithm = args.algorithm if args.mode != "random" else "Random"
    df.to_csv(f"../FinalData/TestLogs/{algorithm}-{args.env_name[4:]}-testing.csv", index=False)
