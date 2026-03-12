import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import Lambda

from torchvision.io import decode_image
from torch.utils.data import DataLoader

device = torch.accelerator.current_accelerator()
print(f"Using {device} device")

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ## Hyperparameters
        self.units = units = 512
        self.learing_rate = 0.001
        self.batch_size = 64
        self.epochs = 5
        self.loss = nn.CrossEntropyLoss()
        self.device = device

        self.model = nn.Sequential(
            ## Parameters
            nn.Linear(784, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 10),
        )

    def forward(self, batch):
        return self.model(batch)

model = NN().to(device)
optim = torch.optim.SGD(model.parameters(), lr=model.learing_rate)

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

        return features, label

one_hot = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
image_dataset = CustomDataset(
    './fashion-mnist/train.csv',
    './fashion-mnist/train/',
    target_transform=one_hot,
)
train_dataloader = DataLoader(image_dataset, batch_size=model.batch_size, shuffle=True)


def train(epoch):
    size = len(image_dataset)
    total = size * model.epochs
    #model.train() # does not actually train.. instead it enables some layers and things to be ready for training.
    for batch, (X, y) in enumerate(train_dataloader):

        ## Get answer from model
        prediction = model(X.to(model.device))

        ## How wrong the models predition was
        loss = model.loss(prediction, y.to(model.device))

        ## Calculage gradients (arrays that can be subtracted to our parameters (layers)  so the model can "LEARN")
        optim.zero_grad()
        loss.backward()

        ## Finally the model can learn
        optim.step()

        ## Print progress
        if batch % 50 == 0:
            progress = f'{100. * (((size * epoch) + (batch * model.batch_size)) / total):.2f}%'
            print(f'{loss:.5f} {progress}')
            #print(batch)

## Train for N epochs
for epoch in range(model.epochs): train(epoch)











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
