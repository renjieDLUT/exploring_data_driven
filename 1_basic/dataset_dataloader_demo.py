import os
import random
import PIL
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# =======================  从torchvision加载数据集 ====================
mnist_train_data = datasets.MNIST("./data/mnist", True, transform=transforms.ToTensor(), download=True)
mnist_test_data = datasets.MNIST("./data/mnist", False, transform=transforms.ToTensor(), download=True)

fashion_train_data = datasets.FashionMNIST(
    "./data/fashion_mnist", True, transform=transforms.ToTensor(), download=True)
fashion_test_data = datasets.FashionMNIST(
    "./data/fashion_mnist", False, transform=transforms.ToTensor(), download=True)

cifar10_train_data = datasets.CIFAR10("./data/cifar10", True, transform=transforms.ToTensor(), download=True)
cifar10_test_data = datasets.CIFAR10("./data/cifar10", False, transform=transforms.ToTensor(), download=True)

for data in [mnist_train_data, fashion_train_data, cifar10_train_data]:
    fig: plt.Figure = plt.figure()
    idx_to_class = {v: k for k, v in data.class_to_idx.items()}
    print(f"{data.__class__.__name__:>12}.shape: {data[0][0].shape}")
    for r in range(3):
        for c in range(3):
            x, y = data[r * 3 + c]
            x = x.permute(1, 2, 0)
            ax: plt.Axes = fig.add_subplot(3, 3, r * 3 + c + 1)
            ax.set_xticks(ticks=[])
            ax.set_yticks(ticks=[])
            ax.set_title(f'{idx_to_class[y]}')
            ax.imshow(x, cmap='gray')
    plt.show()


# =======================  custom dataset ====================

class HymenopteraDataset(Dataset):
    CLASSES = {'ants': 0, 'bees': 1}

    def __init__(self, root: str, train: bool, transform=None):
        super().__init__()
        mode = 'train' if train else 'val'
        root = os.path.join(root, mode)
        subdirs = [dir for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))]

        self.data = []
        for dir in subdirs:
            y = self.CLASSES[dir]
            cur = os.path.join(root, dir)
            imgs = [(os.path.join(cur, img), y) for img in os.listdir(cur) if os.path.isfile(os.path.join(cur, img))]
            self.data.extend(imgs)
        random.shuffle(self.data)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path, label = self.data[item]
        raw_img: PIL.Image = PIL.Image.open(img_path)
        img_np = np.asarray(raw_img)
        img = self.transform(img_np)
        return img, label,raw_img


dataset_path = './data/hymenoptera_data'
my_dataset = HymenopteraDataset(dataset_path,True,transforms.ToTensor())
x,y,z=my_dataset[0]
plt.imshow(z)
plt.show()

def my_collect_fn(data):
    raw_data=data
    data=[d[0:2] for d in raw_data]
    z=[d[2] for d in raw_data]
    return *(torch.utils.data.default_collate(data)),z

# =======================   dataloader ====================
train_dataloader=DataLoader(my_dataset,1,shuffle=True,collate_fn=my_collect_fn)
x,y,z=next(iter(train_dataloader))
print(x.shape,y.shape)
plt.imshow(z[0])
plt.show()




