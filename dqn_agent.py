import torch
import numpy as np
from collections import deque, namedtuple


BUFFER_SIZE = int(1e6)
LR = 5e-4
BATCH_SIZE = 64
TAU = 5e-3
UPDATE_EVERY = 4
GAMMA = 0.99


class DQN_AGENT:
    def __init__(self):
        pass

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
