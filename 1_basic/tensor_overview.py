import torch
t=torch.Tensor([1,2,3])
p=torch.nn.Parameter(t)
print(repr(t),repr(p))