import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

nets = []


# LeNet-5
# 1. cnn
# 2. pooling
# 3. fc
class MyLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x: torch.Tensor):
        feature = self.feature_layer(x)
        return self.classifier(feature.view(x.shape[0], -1))


nets.append(MyLeNet())

# AlexNet
# 1. dropout
# 2. relu
# 3. augmentation ,flip crop, color jit
# 4. LRN
alex_net = torchvision.models.alexnet()
nets.append(alex_net)

# vgg
# 3*3
vgg_net = torchvision.models.vgg16()
nets.append(vgg_net)

# googLeNet
# 1. Inception块
google_net = torchvision.models.GoogLeNet()
nets.append(google_net)

# ResNet
# 1. res
res_net = torchvision.models.resnet18()
nets.append(res_net)


def get_params_num(model: nn.Module):
    num = 0
    for named, param in model.named_parameters():
        num += param.numel()
    return num

for net in nets:
    print('{:>10}:  {:>12,}'.format(net.__class__.__name__,get_params_num(net)))
