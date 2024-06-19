import numpy as np
import torch
import random
from dataclasses import dataclass

#    The PER logic based on: https://arxiv.org/pdf/1511.05952.pdf
#    The tree logic based on: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/replay_buffer.py
#    Proper citations are in the paper [23] [24].


@dataclass
class MemoryItem:
    state: torch.tensor
    action: torch.tensor
    next_state: torch.tensor
    reward: torch.tensor


class PrioritizedExperienceMemory:

    def __init__(self, max_size, alpha, beta, beta_increment):
        # Check if max size is power of 2
        if max_size & (max_size - 1) != 0:
            raise ValueError("Memory size must be power of 2")

        # Initialize memory -- two segment trees for sum and minimum, one array for data
        self.max_size = max_size
        self.priority_sum = np.array([0.0] * (2 * self.max_size))
        self.priority_min = np.array([float("inf")] * (2 * self.max_size))
        self.data = np.array([None] * self.max_size)
        
        # Initialize hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

        # Initialize pointers and index helpers
        self.pointer = 0
        self.current_size = 0

    def append(self, state, action, next_state, reward):
        # Append data to the memory
        self.data[self.pointer] = MemoryItem(state, action, next_state, reward)

        # Update segment trees
        self.update_sum_tree_priorities(self.pointer, self.max_priority ** self.alpha)
        self.update_min_tree_priorities(self.pointer, self.max_priority ** self.alpha)

        # Adjust helper variables -- pointer and current size
        self.current_size = min(self.max_size, self.current_size + 1)
        self.pointer = (self.pointer + 1) % self.max_size

    def sample(self, sample_size):
        # Calculate minimal probability and maximal weight
        minimal_probability = self.priority_min[1] / self.priority_sum[1]
        max_weight = (minimal_probability * self.current_size) ** (-self.beta)

        # Find samples
        pointers = []
        for i in range(sample_size):
            p = random.random() * self.priority_sum[1]
            pointer = self.find_experience_by_priority(p)
            pointers.append(pointer)

        # Find weights
        weights = []
        for i in range(sample_size):
            pointer = pointers[i]
            # Calculate probability and weight based on equations from the paper
            probability = self.priority_sum[pointer + self.max_size] / self.priority_sum[1]
            weight = (probability * self.current_size) ** (-self.beta)
            weights.append(weight / max_weight)

        # Get data
        data = [self.data[pointer] for pointer in pointers]
        return data, weights, pointers

    def adjust_beta(self):
        self.beta = min(1, self.beta + self.beta_increment)

    def __len__(self):
        return self.current_size

    def update_priorities(self, pointers, priorities):
        for pointer, priority in zip(pointers, priorities):
            # Check if new priority is greater than max priority
            self.max_priority = max(self.max_priority, priority)

            # Update segment trees
            self.update_sum_tree_priorities(pointer, priority ** self.alpha)
            self.update_min_tree_priorities(pointer, priority ** self.alpha)

    def update_sum_tree_priorities(self, pointer, priority_alpha):
        # Get leaf pointer, set its priority
        leaf_pointer = pointer + self.max_size
        self.priority_sum[leaf_pointer] = priority_alpha

        # Update priorities until root is reached
        while leaf_pointer >= 2:
            leaf_pointer //= 2
            self.priority_sum[leaf_pointer] = self.priority_sum[2 * leaf_pointer] + self.priority_sum[2 * leaf_pointer + 1]

    def update_min_tree_priorities(self, pointer, priority_alpha):
        # Get leaf pointer, set its priority
        leaf_pointer = pointer + self.max_size
        self.priority_min[leaf_pointer] = priority_alpha

        # Update priorities until root is reached
        while leaf_pointer >= 2:
            leaf_pointer //= 2
            p1 = self.priority_min[2 * leaf_pointer]
            p2 = self.priority_min[2 * leaf_pointer + 1]
            # The linter is not happy with usage of min, so ternary it is
            self.priority_min[leaf_pointer] = p1 if p1 < p2 else p2

    def find_experience_by_priority(self, priority_value):
        # Start at root
        tree_pointer = 1

        # Traverse the tree
        while tree_pointer < self.max_size:
            # If left branch sum is greater than prefix sum, go left
            if self.priority_sum[tree_pointer * 2] > priority_value:
                tree_pointer *= 2
            # If left branch sum is lesser than prefix sum, adjust prefix sum and go right
            else:
                priority_value -= self.priority_sum[tree_pointer * 2]
                tree_pointer = tree_pointer * 2 + 1

        # Return the index of the leaf (subtract max_size to get the index in the data array)
        return tree_pointer - self.max_size
