import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):

    # Network for atari games
    def __init__(self, observation_count, action_count, device):
        super(DuelingDQN, self).__init__()

        # Setup variables
        self.action_count = action_count
        self.observation_count = observation_count
        self.device = device

        # Setup networks -- convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(observation_count, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Setup network -- state value head
        self.value_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Setup network -- advantage head
        self.advantage_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_count)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten

        value = self.value_layers(x)
        advantages = self.advantage_layers(x)

        q_values = value + (advantages - advantages.mean())
        return q_values
