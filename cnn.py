import torch
import torch.nn as nn
import torch.nn.functional as F




class LeNet():
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 6,  5) ## b&w / 6 filters / 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.density = density = 128
        self.linear1 = nn.Linear(16*5,    density)
        self.linear2 = nn.Linear(density, density)
        self.linear3 = nn.Linear(density, 10)

    def forward(self, input):
        out = F.max_pool2d(F.relu(self.conv1(input)), (2,2))






















