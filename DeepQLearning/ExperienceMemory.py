from collections import deque
import random
from dataclasses import dataclass
import torch


@dataclass
class MemoryItem:
    state: torch.tensor
    action: torch.tensor
    next_state: torch.tensor
    reward: torch.tensor


class ExperienceMemory(object):
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def append(self, state, action, next_state, reward):
        self.memory.append(MemoryItem(state, action, next_state, reward))

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
