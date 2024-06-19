import torch
import torch.nn.functional as F
import gymnasium as gym
from DeepQLearning.DQN import DQN
from DeepQLearning.DQNAgent import DQNAgent


class DDQNAgent(DQNAgent):

    def __init__(self,
                 policy_network: DQN,       # Online network that is optimized and used to get actions in DDQN update
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
                 save_dir="DDQN",           # Directory to save the models and .csv reports
                 decay_type="exponential",  # Type of decay for epsilon. Can be linear or exponential
                 linear_decay_regulator=5,  # Used to slow down the linear decay
                 draw_charts=True):         # Draw charts while training

        super().__init__(policy_network, target_network, env, device, epsilon_min=epsilon_min, epsilon_max=epsilon_max,
                         epsilon_decay=epsilon_decay, gamma=gamma, learning_rate=learning_rate, sample_size=sample_size,
                         memory_size=memory_size, tau=tau, training_episodes=training_episodes, max_score=max_score,
                         save_name=save_name, save_interval=save_interval, mean_reward_interval=mean_reward_interval,
                         save_dir=save_dir, decay_type=decay_type, linear_decay_regulator=linear_decay_regulator,
                         draw_charts=draw_charts)

    def advance(self):
        if len(self.memory) < self.sample_size:
            return

        # Sample data and prepare tensors
        samples = self.memory.sample(self.sample_size)
        states, actions, rewards, next_states, done_flags = self.extract_tensors(samples)

        # Compute current Q-values
        current_q_values = self.policy_network(states).gather(1, actions)

        # Get next state values from target network, zero out those that have no following state
        # The selection is done using best actions from policy network, because this is DDQN, not DQN
        with torch.no_grad():
            best_actions = self.policy_network(next_states).max(1).indices.unsqueeze(1)
            next_state_values = self.target_network(next_states).gather(1, best_actions)
            next_state_values[done_flags] = 0.0
            next_state_values = next_state_values.squeeze(1)

        # Compute expected Q-values
        expected_q_values = (next_state_values * self.gamma) + rewards

        # Compute loss between current and expected q values and optimize
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
