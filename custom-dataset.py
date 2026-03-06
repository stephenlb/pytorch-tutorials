import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
import pandas as pd

from torchvision.transforms import ToTensor, Lambda

from torchvision.io import decode_image
from torch.utils.data import DataLoader

labels_array = [
    "T-Shrit",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Bag",
    "Ankel Boot",
]
label_map = {
    l:labels_array[l] for l in range(len(labels_array))
}

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
        features = decode_image(image_path)
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
train_dataloader = DataLoader(image_dataset, batch_size=64, shuffle=True)

def testSample():
    train_features, train_labels = next(iter(train_dataloader))
    print(f'Features: {train_features.size()}')
    print(f'Labels: {train_labels.size()}')
    sample_image = train_features[0].squeeze()
    sample_label = train_labels[0]
    print(f'Label: {sample_label}')

    print(sample_image)
    print(sample_label)
    #plt.imshow(sample_image)
    #plt.show()

testSample()









