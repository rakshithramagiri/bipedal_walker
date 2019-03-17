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
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ts = 0

        self.learning_network = DQ_NETWORK(self.state_size, self.action_size, seed).to(DEVICE)
        self.target_network = DQ_NETWORK(self.state_size, self.action_size, seed).to(DEVICE)
        self.replay_memory = REPLAY_MEMORY(BUFFER_SIZE, BATCH_SIZE, seed)

        self.critertion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.learning_network.parameters(), lr=LR)

    def step(self, state, action, next_state, reward, done):
        self.replay_memory.add(state, action, next_state, reward, done)
        self.ts += 1

        if self.ts % UPDATE_EVERY == 0 and len(self.replay_memory) > BATCH_SIZE:
            experiences = self.replay_memory.sample()
            self.learn(experiences)


    def act(self, state, eps):
        state = torch.from_numpy(state).float().to(DEVICE)
        self.learning_network.eval()
        with torch.no_grad():
            action_values = self.learning_network(state)
        self.learning_network.train()

        if random.random() > eps:
            return action_values.numpy()
        return np.random.choice(np.linspace(-1, 1, 100), size=4)


    def learn(self, experiences):
        states, actions, next_states, rewards, dones = experiences
        targets = self.target_network(torch.from_numpy(next_states))
        targets = rewards + (GAMMA*targets*(1-dones))
        outputs = self.learning_network(torch.from_numpy(states))

        loss = self.critertion(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_network_update()


    def target_network_update(self):
        for target_params, local_params in zip(self.target_network.parameters(), self.learning_network.parameters()):
            target_params.data.copy_(TAU*local_params.data + (1-TAU)*target_params.data)



class REPLAY_MEMORY:
    def __init__(self):
        pass
    def add(self, state, action, next_state, reward, done):
        experience = self.experience(state, action, next_state, reward, done)
        self.memory.append(experience)


    def sample(self):
        pass

    def __len__(self):
        return len(self.memory)
