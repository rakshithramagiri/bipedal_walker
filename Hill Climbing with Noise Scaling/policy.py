import numpy as np


class POLICY:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        np.random.seed(seed)

        self.weights = 1e-4 * np.random.randn(self.state_size, self.action_size)


    def forward(self, state):
        output = np.dot(state, self.weights)
        output = np.exp(output)/np.sum(np.exp(output))
        return output
