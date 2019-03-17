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
        self.learning_network = DQ_NETWORK(self.state_size, self.action_size, seed).to(DEVICE)
        self.target_network = DQ_NETWORK(self.state_size, self.action_size, seed).to(DEVICE)
        self.replay_memory = REPLAY_MEMORY(BUFFER_SIZE, BATCH_SIZE)


    def step(self, state, action, next_state, reward, done):
        self.replay_memory.add(state, action, next_state, reward, done)
        self.ts += 1

        if self.ts % UPDATE_EVERY == 0:
            experiences = self.replay_memory.sample()
            self.learn(experiences)



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
