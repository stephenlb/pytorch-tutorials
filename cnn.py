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
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        out = out.view(-1, self.num_flat_features(out))

        #print
        ## BROKEN HERE< FIX!!!
        #out = F.relu(self.linear1(out))
        #out = F.relu(self.linear2(out))
        #out = self.linear2(out)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:] ## excluding batch size
        features = 1

        for s in size:
            features *= s

        return features

r = torch.rand(1,25,25)
lenet = LeNet()
out = lenet.forward(r)
print(out)
























