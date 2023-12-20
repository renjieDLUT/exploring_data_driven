import torch.nn as nn
import torch

gru = nn.GRU(10, 20, 2, batch_first=True)
input = torch.randn(3, 5, 10)
h0 = torch.randn(2, 3, 20)
output, hn = gru(input, h0)

print(output.shape, hn.shape)
