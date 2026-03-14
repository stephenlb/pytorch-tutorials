import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from torchvision.io import decode_image
from torch.utils.data import DataLoader

X_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

x_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

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
print(label_map)

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3

def plot_samples():
    for i in range(1, cols * rows + 1):
        idx = torch.randint(len(X_data), size=(1,)).item()
        img, label = X_data[idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_array[label])
        plt.axis("off")
        plt.imshow(img.squeeze())#, cmap="gray")
    plt.show()


class CustomDataset(Dataset):
    def __init__(self, csv, image_directory, transform=None, target_transform=None):
        self.labels = pd.read_csv(csv)
        self.image_directory = image_directory
        self.transforms = transforms
        self.target_transform = target_transform
        self.features = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory, self.labels.iloc[index, 0])
        features = decode_image(image_path)
        label = self.labels.iloc[index, 1]

        if self.transforms:
            self.transforms(features)
        if self.target_transform:
            self.target_transform(label)

        return features, label


        

train_dataloader = DataLoader(X_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(x_data, batch_size=64, shuffle=True)

## load 1 data sample and print to the screen the image and label

def testSample():
    train_features, train_labels = next(iter(train_dataloader))
    print(f'Features: {train_features.size()}')
    print(f'Labels: {train_labels.size()}')

    sample_image = train_features[0].squeeze()
    sample_label = train_labels[0]
    print(f'Label: {sample_label}')

    plt.imshow(sample_image)
    plt.show()




























