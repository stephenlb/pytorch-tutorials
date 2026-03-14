import os
import sys
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import Lambda, ToTensor
from torchvision.io import decode_image
from torch.utils.data import DataLoader

## GPU / TPU / CPU
device = torch.accelerator.current_accelerator()
print(f"Using {device} device")

## Class that holds hyperparameters and model weight parameters
class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ## Hyperparameters
        self.units = units = 512
        self.learing_rate = 0.002
        self.batch_size = 128
        self.epochs = 15
        self.loss = nn.CrossEntropyLoss()
        self.device = device
        self.outputs = 10
        self.model = nn.Sequential(
            ## Parameters
            nn.Linear(784, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, self.outputs),
            #nn.Tanh(),
        )

    def forward(self, batch):
        return self.model(batch)

model = NN().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=model.learing_rate)

labels_array = [
    "T-Shrit",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankel Boot",
]

## We never used this even though the tutorial had it.
"""
label_map = {
    l:labels_array[l] for l in range(len(labels_array))
}
"""

class CustomDataset(Dataset):
    def __init__(self, csv, image_directory, transform=None, target_transform=None):
        self.data = pd.read_csv(csv)
        self.image_directory = image_directory
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory, self.data.iloc[index, 1])
        features = decode_image(image_path).to(torch.float).flatten() / 255.0
        label = self.data.iloc[index, 2]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        return features, torch.tensor([label], dtype=torch.float32)

## Flatten 2D image to 1D array
image_transform = Lambda(lambda x: ToTensor()(x).flatten())

## Keep numbers between 0 and 1
## 1 = [1,0,0,0,0,0,0,0,0]
## 2 = [0,1,0,0,0,0,0,0,0]
## ...
## 9 = [0,0,0,0,0,0,0,0,1]
one_hot = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

## Setup image data loading
## Doesn't actually load until we call it from a dataloader
image_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=image_transform,
    target_transform=one_hot,
)
## Test data (we only test the model on this data)
test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=image_transform,
    target_transform=one_hot,
)

## Loader for batching
train_dataloader = DataLoader(image_dataset, batch_size=model.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=model.batch_size, shuffle=True)

def train(epoch):
    ## Vars for printing progress
    size = len(image_dataset)
    total = size * model.epochs

    ## Does not actually train...
    ## Instead it enables layers (Dropout, Norms) for training.
    model.train() 

    ## Load the data into batches for faster GPU training
    for batch, (X, y) in enumerate(train_dataloader):

        ## Get answer from model
        prediction = model(X.to(model.device))

        y = y.to(model.device)

        ## How wrong the models predition was
        loss = model.loss(prediction, y)

        ## Calculage gradients (arrays that can be subtracted to our parameters (layers)  so the model can "LEARN")
        optim.zero_grad()
        loss.backward()

        ## Finally the model can learn
        optim.step()

        ## Print progress
        if batch % 50 == 0:
            progress = f'{100. * (((size * epoch) + (batch * model.batch_size)) / total):.2f}%'
            print(f'Loss: {loss:.5f} {progress}')


## Test the accuracy of our model
@torch.inference_mode
def test():
    model.eval()
    cost = 0 ## precision
    correct = 0
    incorrect = 0 
    for X, y in test_dataloader:
        out = model(X.to(model.device))
        answers = out.argmax(1).cpu()
        loss = model.loss(out, y.to(model.device)).item()
        accuracy = answers == y.argmax(1)
        batch_correct_answers = len(list(filter(lambda a: a, accuracy)))
        correct += batch_correct_answers
        incorrect += model.batch_size - batch_correct_answers
        cost = (cost + loss) / 2

    print(f'Accuracy: {100. * (correct / (correct + incorrect)):.2f}%')
    print(f'Cost: {cost:.2f}')


## Load model if exists, otherwise train and save
model_file = 'fashion.pth'
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, weights_only=True))
else:
    ## Train for N epochs
    for epoch in range(model.epochs): train(epoch)
    torch.save(model.state_dict(), model_file)

## Test
test()


        

## Display some of the image data
def testSample():
    train_features, train_labels = next(iter(train_dataloader))
    print(f'Features: {train_features.size()}')
    print(f'Labels: {train_labels.size()}')
    sample_image = train_features[0].squeeze()
    sample_label = train_labels[0]
    print(f'Label: {sample_label}')

    #print(sample_image)
    #print(sample_label)
    #print(sample_image.shape)
    #print(sample_image.unsqueeze(0).shape)
    #print(nn.Flatten()(sample_image).shape)
    #logits = model(sample_image)
    logits = model(sample_image.to(model.device))
    #print(logits)
    argmax = logits.argmax(0)
    softmax = nn.Softmax(0)(logits)
    print(f"{logits=}")
    print(f"{argmax=}")
    print(f"{softmax=}")
    print(f"{labels_array[argmax]}")
    #plt.imshow(sample_image.reshape((28,28)))
    #plt.show()

#testSample()
