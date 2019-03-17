import torch
import numpy as np
import random
from collections import deque, namedtuple

from model import DQ_NETWORK


BUFFER_SIZE = int(1e6)
LR = 5e-4
BATCH_SIZE = 64
TAU = 5e-3
UPDATE_EVERY = 4
GAMMA = 0.99
DEVICE = 'cpu' # CPU is much faster for simple environments


class DQN_AGENT:
    def __init__(self):
        pass
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ts = 0

    def step(self):
        pass

    def act(self):
        pass

    def learn(self):
        pass

    def target_network_update(self):
        pass


class REPLAY_MEMORY:
    def __init__(self):
        pass

    def add(self):
        pass

    def sample(self):
        pass

    def __len__(self):
        return len(self.memory)
