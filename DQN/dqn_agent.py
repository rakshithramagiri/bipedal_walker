import torch
import numpy as np
import random
from collections import deque, namedtuple

from model import DQ_NETWORK


BUFFER_SIZE = int(1e6)
LR = 5e-2
BATCH_SIZE = 64
TAU = 1e-3
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

        self.critertion = torch.nn.MSELoss()
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
        targets = self.target_network(next_states)
        targets = rewards + (GAMMA*targets*(1-dones))
        outputs = self.learning_network(states)

        loss = self.critertion(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_network_update()


    def target_network_update(self):
        for target_params, local_params in zip(self.target_network.parameters(), self.learning_network.parameters()):
            target_params.data.copy_(TAU*local_params.data + (1-TAU)*target_params.data)



class REPLAY_MEMORY:
    def __init__(self, buffer_size, batch_size, seed):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple('Experience', ['state', 'action', 'next_state', 'reward', 'done'])


    def add(self, state, action, next_state, reward, done):
        experience = self.experience(state, action, next_state, reward, done)
        self.memory.append(experience)


    def sample(self):
        samples = random.choices(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in samples])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in samples])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in samples]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, next_states, rewards, dones)


    def __len__(self):
        return len(self.memory)
