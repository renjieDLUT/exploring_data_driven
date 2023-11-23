import torch
import torchvision.models

bn_layer = torch.nn.BatchNorm2d(16)
for n, p in bn_layer.named_parameters():
    print(n, p.shape)
torchvision.models.resnet50()

ln_layer = torch.nn.LayerNorm([3, 28, 28])

for n, p in ln_layer.named_parameters():
    print(n, p.shape)

dropout = torch.nn.Dropout(1.)
for n, p in dropout.named_parameters():
    print(n, p.shape)

x = torch.ones(10)
w = torch.ones(10, requires_grad=True)

y = x * w
y = dropout(y).sum()
print(w.grad)

print(w.grad)
