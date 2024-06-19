import gymnasium as gym
import gymnasium.wrappers as wrappers
import torch
import numpy as np
import argparse as ap
from DeepQLearning.DQN import DQN
from DeepQLearning.DQNAgent import DQNAgent
from DeepQLearning.DDQNAgent import DDQNAgent
from Utils.Printer import Printer
from DeepQLearning.DuelingDQN import DuelingDQN
from DeepQLearning.DuelingPDQNAgent import DuelingPDQNAgent

# Set and parse arguments
parser = ap.ArgumentParser(description="Deep Q Learning, Double Deep Q Learning and Dueling Deep Q-Learning with "
                                       "Prioritized Replay implementation -- Train an agent to"
                                       "play Atari games. All arguments are optional, default values are set.")
parser.add_argument("-algorithm", type=str, default="DQN", help="Algorithm to use. Can be DQN or DDQN or DuelingPDQN.")
parser.add_argument("-env_name", type=str, default="ALE/Pong-v5", help="Environment name.")
parser.add_argument("-epsilon_min", type=float, default=0.05, help="Minimum epsilon value for epsilon-greedy action "
                                                                   "selection. Must be in range (0, 1).")
parser.add_argument("-epsilon_max", type=float, default=0.99, help="Maximum epsilon value for epsilon-greedy action "
                                                                  "selection. Must be in range (0, 1).")
parser.add_argument("-epsilon_decay", type=float, default=0.00025, help="Epsilon decay value for epsilon-greedy action "
                                                                      "selection. Tiny number (0.000x) is recommended.")
parser.add_argument("-gamma", type=float, default=0.99, help="Discount factor for future rewards.")
parser.add_argument("-learning_rate", type=float, default=0.00025, help="Learning rate for the optimizer.")
parser.add_argument("-sample_size", type=int, default=128, help="Sample size for experience memory.")
parser.add_argument("-memory_size", type=int, default=131072, help="Memory size for experience memory. Must be power of 2.")
parser.add_argument("-tau", type=float, default=0.005, help="Tau value for soft update of target network.")
parser.add_argument("-training_episodes", type=int, default=1000, help="Number of episodes to train the agent.")
parser.add_argument("-max_score", type=int, default=-21, help="Maximum score to save the network. Can't be positive.")
parser.add_argument("-save_name", type=str, default="Pong-episode-", help="Name of the saved network.")
parser.add_argument("-save_interval", type=int, default=100, help="Interval to save the network.")
parser.add_argument("-mean_reward_interval", type=int, default=25, help="Interval to calculate mean reward.")
parser.add_argument("-decay_type", type=str, default="exponential", help="Type of decay for epsilon. Can be linear or "
                                                                         "exponential. ")
parser.add_argument("-linear_decay_regulator", type=int, default=5, help="Regulator for linear decay. Divides the "
                                                                         "number of episodes used in epsilon "
                                                                         "calculation.")
parser.add_argument("-draw_charts", type=str, default="True", help="Draw charts while training.")
parser.add_argument("-beta", type=float, default=0.4, help="Initial bias correction coefficient for PER.")
parser.add_argument("-beta_increment", type=float, default=0.001, help="Beta increment for PER.")
parser.add_argument("-alpha", type=float, default=0.5, help="Prioritization coefficient for PER.")
args = parser.parse_args()

# Validate arguments
if not 0 < args.epsilon_min < 1:
    Printer.print_error("Epsilon min must be in range (0, 1)!")
    exit(1)
if not 0 < args.epsilon_max < 1:
    Printer.print_error("Epsilon max must be in range (0, 1)!")
    exit(1)
if args.gamma <= 0:
    Printer.print_error("Gamma must be greater than 0!")
    exit(1)
if args.learning_rate <= 0:
    Printer.print_error("Learning rate must be greater than 0!")
    exit(1)
if args.sample_size <= 0:
    Printer.print_error("Sample size must be greater than 0!")
    exit(1)
if args.memory_size <= 0:
    Printer.print_error("Memory size must be greater than 0!")
    exit(1)
if args.tau <= 0:
    Printer.print_error("Tau must be greater than 0!")
    exit(1)
if args.training_episodes <= 0:
    Printer.print_error("Training episodes must be greater than 0!")
    exit(1)
if args.max_score > 0:
    Printer.print_error("Max score must be less than 0!")
    exit(1)
if args.save_interval <= 0:
    Printer.print_error("Save interval must be greater than 0!")
    exit(1)
if args.mean_reward_interval <= 0:
    Printer.print_error("Mean reward interval must be greater than 0!")
    exit(1)
if args.decay_type not in ["linear", "exponential"]:
    Printer.print_error("Decay type must be linear or exponential!")
    exit(1)
if args.linear_decay_regulator <= 0:
    Printer.print_error("Linear decay regulator must be greater than 0!")
    exit(1)
if args.draw_charts not in ["True", "False"]:
    Printer.print_error("Draw charts must be True or False!")
    exit(1)
if args.beta < 0 or args.beta > 1:
    Printer.print_error("Beta must be in range (0, 1)!")
    exit(1)
if args.alpha < 0 or args.alpha > 1:
    Printer.print_error("Alpha must be in range (0, 1)!")
    exit(1)
if args.beta_increment <= 0:
    Printer.print_error("Beta increment must be greater than 0!")
    exit(1)

# Init device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    Printer.print_success("CUDA is available, using GPU.")
else:
    Printer.print_error("CUDA is not available, using CPU! This will result in very slow training!")

# Init environment and its wrappers.
# AtariWrapper -> Converts image to grayscale, crops useless parts of the screen and resizes it to 84x84
# FrameStack -> Stack 4 frames together, so the network can see the movement of ball, paddles and stuff...
env: gym.Env = gym.make(args.env_name, repeat_action_probability=0.1)
env: gym.Env = wrappers.AtariPreprocessing(env, frame_skip=1)
env: gym.Env = wrappers.FrameStack(env, 4)
Printer.print_success(f"Environment {args.env_name} successfully initialized.")

# Restart environment, init action and observation count
state: np.ndarray = env.reset()[0]
action_count = env.action_space.n
observation_count = len(state)
Printer.print_success(f"Action count: {action_count}\nObservation count: {observation_count}\n")

# Init networks
policy_net: DQN = DQN(observation_count, action_count, device).to(device)
target_net: DQN = DQN(observation_count, action_count, device).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Setup agent with networks, env, device and args
args.algorithm = args.algorithm.upper()
if args.algorithm == "DQN":
    Printer.print_info(f"Using DQN algorithm with {args.decay_type} decay...")
    agent: DQNAgent = DQNAgent(policy_net, target_net, env, device, epsilon_min=args.epsilon_min,
                               epsilon_max=args.epsilon_max, epsilon_decay=args.epsilon_decay, gamma=args.gamma,
                               learning_rate=args.learning_rate, sample_size=args.sample_size,
                               memory_size=args.memory_size,
                               tau=args.tau, training_episodes=args.training_episodes, max_score=args.max_score,
                               save_name=args.save_name, save_interval=args.save_interval,
                               mean_reward_interval=args.mean_reward_interval, decay_type=args.decay_type,
                               linear_decay_regulator=args.linear_decay_regulator, draw_charts=args.draw_charts)
    print(type(agent))
elif args.algorithm == "DDQN":
    Printer.print_info(f"Using DDQN algorithm with {args.decay_type} decay...")
    agent: DDQNAgent = DDQNAgent(policy_net, target_net, env, device, epsilon_min=args.epsilon_min,
                                 epsilon_max=args.epsilon_max, epsilon_decay=args.epsilon_decay, gamma=args.gamma,
                                 learning_rate=args.learning_rate, sample_size=args.sample_size,
                                 memory_size=args.memory_size, tau=args.tau, training_episodes=args.training_episodes,
                                 max_score=args.max_score, save_name=args.save_name, save_interval=args.save_interval,
                                 mean_reward_interval=args.mean_reward_interval, decay_type=args.decay_type,
                                 linear_decay_regulator=args.linear_decay_regulator, draw_charts=args.draw_charts)
    print(type(agent))
elif args.algorithm == "DUELINGPDQN":
    policy_net: DuelingDQN = DuelingDQN(observation_count, action_count, device).to(device)
    target_net: DuelingDQN = DuelingDQN(observation_count, action_count, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    Printer.print_info(f"Using Dueling DQN with Prioritized Replay algorithm with {args.decay_type} decay...")
    agent: DuelingPDQNAgent = DuelingPDQNAgent(policy_net, target_net, env, device, epsilon_min=args.epsilon_min,
                                               epsilon_max=args.epsilon_max, epsilon_decay=args.epsilon_decay,
                                               gamma=args.gamma, learning_rate=args.learning_rate, sample_size=args.sample_size,
                                               memory_size=args.memory_size, tau=args.tau, training_episodes=args.training_episodes,
                                               max_score=args.max_score, save_name=args.save_name, save_interval=args.save_interval,
                                               mean_reward_interval=args.mean_reward_interval, decay_type=args.decay_type,
                                               linear_decay_regulator=args.linear_decay_regulator, draw_charts=args.draw_charts,
                                               alpha=args.alpha, beta=args.beta, beta_increment=args.beta_increment)
    print(type(agent))
else:
    Printer.print_error("Algorithm must be DQN, DDQN or DuelingPDQN!")
    exit(1)

# Training loop
agent.train()

# Save networks, finish training
Printer.print_success("\nTraining finished!")
Printer.print_success("Saving network...")
torch.save(agent.target_network.state_dict(), f"Models/{args.algorithm}/{args.save_name}final.pth")
Printer.print_success("Network saved.")
input("Press key to end the program...")
