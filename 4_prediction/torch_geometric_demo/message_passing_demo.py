from torch_geometric.utils.softmax import softmax
import torch
import torch.nn as nn

src = torch.tensor([1., 1., 1., 1.])
index = torch.tensor([0, 0, 0, 3])
print(softmax(src, index))
fn=nn.Softmax()
print(fn(torch.tensor([0.,0.,1.,1.])))