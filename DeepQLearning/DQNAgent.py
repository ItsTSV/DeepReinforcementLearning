import math
import random
import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from itertools import count
from DeepQLearning.DQN import DQN
from DeepQLearning.ExperienceMemory import ExperienceMemory
from Utils.Printer import Printer
from Utils.Monitoring import Monitoring


class DQNAgent:

    def __init__(self,
                 policy_network: DQN,       # Online network that is being optimized during training
                 target_network: DQN,       # Target network that is used to compute target Q-values
                 env: gym.Env,              # Environment to train on
                 device: torch.device,      # Device to run the training on
                 epsilon_min=0.05,          # Minimum epsilon value for epsilon greedy action selection
                 epsilon_max=0.99,          # Maximum epsilon value for epsilon greedy action selection
                 epsilon_decay=0.00025,     # Epsilon decay value for epsilon greedy action selection
                 gamma=0.99,                # Discount factor (1: fully prefer value, 0: fully prefer immediate reward)
                 learning_rate=0.00025,     # Learning rate for Adam optimizer
                 sample_size=128,           # How many samples from memory are used in each training step
                 memory_size=131072,        # How many samples can be stored in memory
                 tau=0.005,                 # Soft update parameter for target network
                 training_episodes=1000,    # How many episodes to train the agent
                 max_score=-21,             # Maximum score to save the network. Can't be positive
                 save_name="Game-episode",  # Name of the saved network
                 save_interval=100,         # Interval to save the network weights
                 mean_reward_interval=50,   # Interval to calculate mean reward (and save the weights if it's the best)
                 save_dir="DQN",            # Directory to save the models and .csv reports
                 decay_type="exponential",  # Type of decay for epsilon. Can be linear or exponential
                 linear_decay_regulator=5,  # Used to slow down the linear decay
                 draw_charts=True):         # Draw charts while training

        # Setup networks
        self.policy_network = policy_network
        self.target_network = target_network

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=learning_rate, amsgrad=True)

        # Setup device
        self.device = device

        # Setup memory
        self.memory = ExperienceMemory(memory_size)
        self.sample_size = sample_size

        # Setup hyperparameters -- epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.epsilon_counter = 0
        self.epsilon = epsilon_max
        self.decay_type = decay_type
        self.linear_decay_regulator = linear_decay_regulator

        # Setup hyperparameters -- neural networks
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau

        # Training parameters
        self.env = env
        self.training_episodes = training_episodes

        # Monitoring parameters
        self.monitoring = Monitoring(mean_reward_interval=mean_reward_interval, save_dir=save_dir, save_name=save_name)
        self.save_name = save_name
        self.max_score = max_score
        self.save_interval = save_interval
        self.mean_reward_interval = mean_reward_interval
        self.save_dir = save_dir
        self.draw_charts = False if draw_charts == "False" else True

        # Print info
        Printer.print_info(f"Initialized with training parameters:\nEpsilon min: {self.epsilon_min}\nEpsilon max: {self.epsilon_max}\n"
                           f"Epsilon decay: {self.epsilon_decay}\nGamma: {self.gamma}\nLearning rate: {self.learning_rate}\n"
                           f"Sample size: {self.sample_size}\nMemory size: {memory_size}\nTau: {self.tau}"
                           f"\nTraining episodes: {self.training_episodes}")

    def epsilon_greedy_action(self, state: torch.tensor):
        # True -> Exploit and choose the best action, False -> Explore and choose a random action
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1)
        else:
            return torch.randint(0, self.policy_network.action_count, (1, 1), device=self.device, dtype=torch.int64)

    def best_action(self, state: torch.tensor):
        with torch.no_grad():
            return self.policy_network(state).max(1)[1].view(1, 1)

    def random_action(self):
        return torch.randint(0, self.policy_network.action_count, (1, 1), device=self.device, dtype=torch.int64)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            if self.decay_type == "exponential":
                self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(
                    -self.epsilon_counter * self.epsilon_decay)
            elif self.decay_type == "linear":
                self.epsilon = max(self.epsilon_min, self.epsilon_max - self.epsilon_counter *
                                   self.epsilon_decay / self.linear_decay_regulator)
        self.epsilon_counter += 1

    def soft_tau_update(self):
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def advance(self):
        if len(self.memory) < self.sample_size:
            return

        # Sample data and prepare tensors
        samples = self.memory.sample(self.sample_size)
        states, actions, rewards, next_states, done_flags = self.extract_tensors(samples)

        # Compute current Q-values
        current_q_values = self.policy_network(states).gather(1, actions)

        # Get next state values from target network, zero out those that have no following state
        with torch.no_grad():
            next_state_values = self.target_network(next_states)
            next_state_values[done_flags] = 0.0
            next_state_values = next_state_values.max(1).values

        # Compute expected Q-values
        expected_q_values = (next_state_values * self.gamma) + rewards

        # Compute loss between current and expected q values and optimize
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def extract_tensors(self, samples):
        states = torch.cat([sample.state for sample in samples])
        actions = torch.cat([sample.action for sample in samples])
        rewards = torch.cat([sample.reward for sample in samples])
        next_states = torch.cat([sample.next_state for sample in samples if sample.next_state is not None])
        done_flags = torch.tensor([sample.next_state is None for sample in samples],
                                  device=self.device, dtype=torch.bool)
        return states, actions, rewards, next_states, done_flags

    def train(self):
        for episode in range(self.training_episodes):
            Printer.print_success(f"\nStarting episode {episode}")

            # Reset the environment and get it's state
            state = self.env.reset()[0]
            state = np.array(state, dtype=np.float32)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Reset the total reward for this episode
            episode_reward = 0

            for step_count in count():
                # Select epsilon greedy action, step environment and get resulting params
                action = self.epsilon_greedy_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())

                # Convert data from environment to tensors
                reward_tensor = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                next_state = np.array(next_state, dtype=np.float32)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store episode in agent memory, update reward and state
                self.memory.append(state, action, next_state, reward_tensor)
                episode_reward += reward
                state = next_state

                # Update epsilon, and advance the model, update tau
                self.update_epsilon()
                self.advance()
                self.soft_tau_update()

                # If episode is done, save every nth episode weights
                if done:
                    Printer.print_success(
                        f"Episode {episode} finished after {step_count + 1} steps with score {episode_reward} (epsilon = {self.epsilon})")
                    self.monitoring.add_data(episode, episode_reward, step_count + 1)
                    if self.draw_charts:
                        self.monitoring.plot_steps()
                        self.monitoring.plot_rewards()

                    if len(self.monitoring.reward_history) >= self.mean_reward_interval and np.mean(
                            self.monitoring.reward_history[-self.mean_reward_interval:]) > self.max_score:
                        self.max_score = np.mean(self.monitoring.reward_history[-self.mean_reward_interval:])
                        torch.save(self.target_network.state_dict(), f"Models/{self.save_dir}/{self.save_name}Max.pth")
                        Printer.print_info(f"Saving model with new max score over last {self.mean_reward_interval} episodes: {self.max_score}")

                    if episode % self.save_interval == 0:
                        torch.save(self.target_network.state_dict(), f"Models/{self.save_dir}/{self.save_name}{episode}.pth")
                    break
