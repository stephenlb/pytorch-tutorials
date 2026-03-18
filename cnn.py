import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        ## Hyperparameters
        self.batch_size = 4
        self.epchos = 10
        self.learning_rate = 0.002
        self.density = density = 120
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = len(self.classes)

        ## Training dataset
        self.trainset = None
        self.dataloader = None

        ## Load data and create Transform for Vectorize to 0 and 1 range
        self.loadData()

        self.conv1   = nn.Conv2d(1, 6,  5) ## b&w / 6 filters / 5x5
        self.conv2   = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16*5*5,  density)
        self.linear2 = nn.Linear(density, density)
        self.linear3 = nn.Linear(density, self.num_classes)

    def loadData(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #mean=tensor([0.4914, 0.4822, 0.4465])
            #std=tensor([0.2470, 0.2435, 0.2616])
            self.normVecs(),
        ])
        self.onehot = transforms.Lambda( lambda x: torch.tensor([x==class for x in range(self.num_classes)], dtype=torch.float32))
        
        ## Training Data
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data',
            download=True,
            train=True,
            transform=self.transform,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=4,
            shuffle=True,
        )

        ## Test Data
        self.testset = torchvision.datasets.CIFAR10(
            root='./data',
            download=True,
            train=False,
            transform=self.transform,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=4,
            shuffle=True,
        )
        
    def normVecs(self):
        transform = transforms.Compose([transforms.ToTensor(),])
        trainset = torchvision.datasets.CIFAR10(
            root='./data',
            download=True,
            train=True,
            transform=transform,
        )
        out  = torch.stack([s[0] for s in ConcatDataset([trainset])])
        std  = torch.std(out, dim=(0,2,3)) 
        mean = torch.mean(out, dim=(0,2,3)) 

        return transforms.Normalize(mean, std)

    def showImage(self):
        iterator = iter(self.dataloader)
        images, labels = next(iterator)

        images  = torchvision.utils.make_grid(images)
        images  = images / 2 + 0.5
        npimage = images.numpy()
        #npimage = np.mean(npimage, axis=2)
        plt.imshow(np.transpose(npimage, (1, 2, 0)))
        print(labels)
        print([self.classes[l] for l in labels])
        plt.show()

    def forward(self, input):
        out = F.max_pool2d(F.relu(self.conv1(input)), (2,2))
        out = F.max_pool2d(F.relu(self.conv2(out)), 2)
        out = out.view(-1, self.num_flat_features(out))

        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)

        return out

    ## "Optimization phase" train the model
    def train(self):
        pass

    def num_flat_features(self, out):
        size = out.size()[1:] ## excluding batch size
        features = 1

        for s in size:
            features *= s

        return features

##  B = Batch
##  C = Color (RGBAXYZ)
##  W = Width
##  H = Height
##             B  C  W   H
r = torch.rand(1, 1, 32, 32)
lenet = LeNet()
lenet.showImage()
#print(r)
#print(lenet)
out = lenet(r)
#print(out)
#print(lenet.transform)








