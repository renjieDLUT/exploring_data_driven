import torch
import torch.nn as nn
import math

x = torch.ones(1, 100, 256)
transformer = nn.MultiheadAttention(256, 8, batch_first=True)
y = transformer(x, x, x, need_weights=False)
print(y[0].shape)

q = torch.ones(1, 100, 256)
k = torch.ones(1, 10, 100)
v = torch.ones(1, 10, 100)
transformer = nn.MultiheadAttention(256, num_heads=8, batch_first=True, kdim=100, vdim=100)
y = transformer(q, k, v, need_weights=False)
print("shape:", y[0].shape)


class MySelfAttention(nn.Module):
    def __init__(self, dim, dk, dv):
        super().__init__()
        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn: torch.Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        result = attn @ v
        return result


my_att = MySelfAttention(256, 256, 256)
y = my_att(x)
print(y.shape)


class MyMultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim=None, num_heads=1):
        super().__init__()
        if dim == None:
            dim = dim_in
        self.dim_in = dim_in
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.q = nn.Linear(dim_in, dim)
        self.k = nn.Linear(dim_in, dim)
        self.v = nn.Linear(dim_in, dim)

        self.scale = 1 / math.sqrt(dim // num_heads)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        bs, length, dim_in = x.shape
        assert dim_in == self.dim_in
        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        nh = self.num_heads
        dk = self.dim // self.num_heads

        q = q.reshape(bs, length, nh, dk).transpose(1, 2)  # (bs,nh,length,dk)
        k = k.reshape(bs, length, nh, dk).transpose(1, 2)
        v = k.reshape(bs, length, nh, dk).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) * self.scale  # (bs, nh, length, length)
        attn = attn.softmax(dim=-1)
        att = (attn @ v).transpose(1, 2).reshape(bs, length,
                                                 self.dim)  # (bs, nh, length, dk) ->  (bs, length, nh, dk) (bs, length, dim)

        result = self.fc(att)
        return result


my_multi_head = MyMultiHeadAttention(256, 256, 8)
y = my_multi_head(x)
print(y.shape)
