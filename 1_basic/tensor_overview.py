import torch

t = torch.Tensor([1, 2, 3])
p = torch.nn.Parameter(t)
print(repr(t), repr(p))

x = t
w = torch.ones(3, requires_grad=True)
b = torch.ones(3, requires_grad=True)

y = x * w + b
print(y.shape, w.grad)
y=y.sum()


y=y.detach()
y=y+3
y1=y.clone()

