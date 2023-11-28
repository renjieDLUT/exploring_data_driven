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


# depthwise separable convolution
class DepthWiseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, has_identity: bool = True):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_dim, in_dim, 3, 1, padding=1, padding_mode="replicate", groups=in_dim)
        self.bn_1 = nn.BatchNorm2d(in_dim)
        self.pool_1 = nn.AvgPool2d(3, 2, padding=1)

        self.conv_2 = nn.Conv2d(in_dim, out_dim, 1, 1)
        self.bn_2 = nn.BatchNorm2d(out_dim)
        self.pool_2 = nn.MaxPool2d(3, 2, padding=1)

        self.has_identity = has_identity
        self.conv_identity = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        raw_x = x
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        # x = self.pool_1(x)

        y = self.conv_2(x)
        y = self.pool_2(y)

        if self.has_identity:
            y += self.conv_identity(raw_x)

        y = self.bn_2(y)
        y = nn.functional.relu(y)
        return y


class MyDepthWiseModule(nn.Module):
    def __init__(self, image_size=32):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, padding_mode="replicate"),
                                     nn.BatchNorm2d(16),
                                     DepthWiseLayer(16, 64),
                                     DepthWiseLayer(64, 256),
                                     DepthWiseLayer(256, 1024),
                                     nn.BatchNorm2d(1024),
                                     )

        self.pool=nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        feature = self.feature(x)
        feature=self.pool(feature)
        feature = feature.reshape(x.shape[0], -1)
        out = self.classifier(feature)
        return out


nets.append(MyDepthWiseModule())


def get_params_num(model: nn.Module):
    num = 0
    for named, param in model.named_parameters():
        num += param.numel()
    return num


for net in nets:
    print('{:>10}:  {:>12,}'.format(net.__class__.__name__, get_params_num(net)))
