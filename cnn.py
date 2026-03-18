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
        self.batch_size = 128
        self.epochs = 10
        self.learning_rate = 0.001
        self.density = density = 256
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = len(self.classes)
        self.loss = nn.CrossEntropyLoss()

        ## Training dataset
        self.trainset = None
        self.trainloader = None

        ## Load data and create Transform for Vectorize to 0 and 1 range
        self.loadData()

        self.conv1   = nn.Conv2d(3, 6,  5) ## RGB / 6 filters / 5x5
        self.conv2   = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16*5*5,  density)
        self.linear2 = nn.Linear(density, density)
        self.linear3 = nn.Linear(density, self.num_classes)

    ## Ask the AI to answer based on input
    def forward(self, input):
        out = F.max_pool2d(F.relu(self.conv1(input)), (2,2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2,2))
        out = out.view(-1, self.num_flat_features(out))

        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)

        return out

    def loadData(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #mean=tensor([0.4914, 0.4822, 0.4465])
            #std=tensor([0.2470, 0.2435, 0.2616])
            self.normVecs(),
        ])
        #one_hot = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        self.onehot = transforms.Lambda(
            lambda x: torch.tensor([
                x == label for label in range(self.num_classes)
            ], dtype=torch.float32)
        )
        
        ## Training Data
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data',
            download=True,
            train=True,
            transform=self.transform,
            target_transform=self.onehot,
        )
        self.trainloader = torch.utils.data.DataLoader(
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
            target_transform=self.onehot,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=4,
            shuffle=True,
        )
        
    def normVecs(self):
        #transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(
            root='./data',
            download=True,
            train=True,
            transform=transforms.ToTensor(),
            #dtype=torch.float32,
        )
        out  = torch.stack([s[0] for s in ConcatDataset([trainset])])
        std  = torch.std(out, dim=(0,2,3)) 
        mean = torch.mean(out, dim=(0,2,3)) 

        return transforms.Normalize(mean, std)

    def showImage(self):
        iterator = iter(self.trainloader)
        images, labels = next(iterator)

        print(labels)
        labels = torch.argmax(labels, axis=1)
        images  = torchvision.utils.make_grid(images)
        images  = images / 2 + 0.5
        npimage = images.numpy()
        #npimage = np.mean(npimage, axis=2)
        plt.imshow(np.transpose(npimage, (1, 2, 0)))
        print(labels)
        print([self.classes[l] for l in labels])
        plt.show()


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
optim = torch.optim.AdamW(lenet.parameters(), lr=lenet.learning_rate)#, momentum=0.9)
device = torch.accelerator.current_accelerator()

## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
## "Optimization phase" train the model
## The model will begin to learn here!
## =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
lenet.to(device)
lenet.train()
cost = 0.0
losses = []
batch = 0
for epoch in range(lenet.epochs):
    for images, labels in lenet.trainloader:
        optim.zero_grad()
        out = lenet(images.to(device))
        loss = lenet.loss(out, labels.to(device))

        ## Calculat Gradients
        loss.backward()

        ## Apply gradients to model weights
        optim.step()

        losses.append(loss)
        #cost = (cost + loss) / 2.
        batch += 1
        if not(batch % 100):
            print(f'Cost: {sum(losses)/float(len(losses)):.2f} Epoch: {epoch+1}')

            
#lenet.showImage()
#print(r)
#print(lenet)
#out = lenet(r)
#print(out)
#print(lenet.transform)








