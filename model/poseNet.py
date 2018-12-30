"""
Net for pose loss
Output score for each part
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

class poseNet(nn.Module):
    def __init__(self, indim, outdim):
        super(poseNet, self).__init__()
        self.fc1 = nn.Linear(indim, 2*indim)
        self.fc2 = nn.Linear(2 * indim, 2 * indim)
        self.fc3 = nn.Linear(2 * indim, outdim)

    def forward(self, x):
        y = self.fc1(x)
        y = functional.relu(y)
        y = self.fc2(y)
        y = functional.relu(y)
        y = self.fc3(y)
        y = 1 / (1 + torch.exp(y))
        return y


