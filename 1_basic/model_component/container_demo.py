import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


net = Model()
print('net.training:', net.training)

net.add_module("conv3", nn.Conv3d(20, 10, 3))


@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)


net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)
net.bfloat16()
print("*" * 30, "buffer", "*" * 30)
for buf in net.buffers():
    print(type(buf), buf.size())

print("*" * 30, "children", "*" * 30)
for child in net.children():
    print(type(child), child)

print("*" * 30, "nn.Sequential", "*" * 30)
net = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)
net = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 20, 5)),
    ('relu1', nn.ReLU())
]))


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
        self.relu1=nn.ReLU()
        self.conv1=nn.Conv2d(1,3,4)
        self.conv2=nn.Conv2d(3,8,3)

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = self.linear(x)
            return x
net=MyModule()
print("*" * 30, "children", "*" * 30)
for child in net.children():
    print(type(child), child)

from torchvision.models import resnet50
net=resnet50()
print("*" * 30, "resnet50 children", "*" * 30)
for child in net.children():
    print(type(child), child)