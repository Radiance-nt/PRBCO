import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(Net, self).__init__()
        self.mlps = nn.Sequential(
            nn.Linear(state_space_size, 40),
            nn.ReLU(),

            # nn.Linear(40, 80),
            # nn.ReLU(),
            #
            # nn.Linear(80, 120),
            # nn.ReLU(),
            #
            # nn.Linear(120, 100),
            # nn.ReLU(),
            #
            # nn.Linear(100, 40),
            # nn.ReLU(),

            nn.Linear(40, 20),
            nn.ReLU(),

            nn.Linear(20, action_space_size),
            nn.Softmax()
        )

    def forward(self, x):
        output = self.mlps(x)
        return output
