import torch
import torch.nn.functional as F
import gymnasium as gym
from DeepQLearning.DuelingDQN import DuelingDQN
from DeepQLearning.DQNAgent import DQNAgent
from DeepQLearning.PrioritizedExperienceMemory import PrioritizedExperienceMemory
from Utils.Printer import Printer


class DuelingPDQNAgent(DQNAgent):

    def __init__(self,
                 policy_network: DuelingDQN,   # Online network that is being optimized during training
                 target_network: DuelingDQN,   # Target network that is used to compute target Q-values
                 env: gym.Env,                 # Environment to train on
                 device: torch.device,         # Device to run the training on
                 epsilon_min=0.05,             # Minimum epsilon value for epsilon greedy action selection
                 epsilon_max=0.99,             # Maximum epsilon value for epsilon greedy action selection
                 epsilon_decay=0.00025,        # Epsilon decay value for epsilon greedy action selection
                 gamma=0.99,                   # Discount factor (1: fully prefer value, 0: fully prefer immediate reward)
                 learning_rate=0.00025,        # Learning rate for Adam optimizer
                 sample_size=128,              # How many samples from memory are used in each training step
                 memory_size=131072,           # How many samples can be stored in memory
                 tau=0.005,                    # Soft update parameter for target network
                 training_episodes=1000,       # How many episodes to train the agent
                 max_score=-21,                # Maximum score to save the network. Can't be positive
                 save_name="Game-episode",     # Name of the saved network
                 save_interval=100,            # Interval to save the network weights
                 mean_reward_interval=50,      # Interval to calculate mean reward (and save the weights if it's the best)
                 save_dir="DuelingPDQN",       # Directory to save the models and .csv reports
                 decay_type="exponential",     # Type of decay for epsilon. Can be linear or exponential
                 linear_decay_regulator=5,     # Used to slow down the linear decay
                 draw_charts=True,             # Draw charts while training
                 alpha=0.5,                    # Prioritization coefficient (how much prioritization is used?)
                 beta=0.4,                     # Initial bias correction coefficient
                 beta_increment=0.001):        # Beta increases over time; how much per step?

        super().__init__(policy_network, target_network, env, device, epsilon_min=epsilon_min, epsilon_max=epsilon_max,
                         epsilon_decay=epsilon_decay, gamma=gamma, learning_rate=learning_rate, sample_size=sample_size,
                         memory_size=memory_size, tau=tau, training_episodes=training_episodes, max_score=max_score,
                         save_name=save_name, save_interval=save_interval, mean_reward_interval=mean_reward_interval,
                         save_dir=save_dir, decay_type=decay_type, linear_decay_regulator=linear_decay_regulator,
                         draw_charts=draw_charts)

        # Override memory with prioritized experience memory
        self.memory = PrioritizedExperienceMemory(memory_size, alpha=alpha, beta=beta, beta_increment=beta_increment)

        # Print additional parameters
        Printer.print_info(f"Prioritization coefficient alpha: {alpha}")
        Printer.print_info(f"Initial bias correction beta: {beta}")
        Printer.print_info(f"Beta increment per step: {beta_increment}")

    def advance(self):
        if len(self.memory) < self.sample_size:
            return

        # Sample data and prepare tensors
        samples, weights, pointers = self.memory.sample(self.sample_size)
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

        # Compute loss between current and expected q values, weighted by priorities
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1), reduction='none')
        weighted_loss = (torch.tensor(weights, device=self.device).unsqueeze(1) * loss).mean()

        # Update priorities in memory (add small value to avoid zero priorities)
        priorities = loss.detach().squeeze().cpu().numpy() + 1e-5
        self.memory.update_priorities(pointers, priorities)

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Adjust beta
        self.memory.adjust_beta()
