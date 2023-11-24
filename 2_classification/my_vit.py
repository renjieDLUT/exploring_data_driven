import torch
import torchvision.models
from torch import nn as nn
from collections import OrderedDict
import math
import torch.utils.data
import time


class MyEncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, norm_layer, dropout, attention_dropout):
        super(MyEncoderBlock, self).__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = torchvision.ops.MLP(in_channels=hidden_dim, hidden_channels=[mlp_dim, hidden_dim],
                                       activation_layer=nn.GELU, dropout=dropout)

    def forward(self, input):
        assert input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class MyVITEncoder(nn.Module):
    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer,
    ):
        super(MyVITEncoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = MyEncoderBlock(num_heads, hidden_dim, mlp_dim, norm_layer, dropout=dropout,
                                                          attention_dropout=attention_dropout)

        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input):
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class MyVIT(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, hidden_dim: int = 768, mlp_dim: int = 3072,
                 attention_dropout: float = 0.0, dropout: float = 0.0, num_classes: int = 1000,
                 num_layer=12, num_heads=3):
        super(MyVIT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = self.patch_size ** 2 * 3 if hidden_dim is None else hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes

        # self.conv_proj1 = nn.Conv2d(in_channels=3, out_channels=self.hidden_dim, kernel_size=patch_size,
        #                             stride=patch_size)
        self.conv_proj1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                                    stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv_proj2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3,
                                    stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_proj3 = nn.Conv2d(64, self.hidden_dim, kernel_size=patch_size, stride=patch_size)

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length = (image_size // patch_size) ** 2

        self.seq_length = seq_length + 1

        self.encoder = MyVITEncoder(self.seq_length, num_layer, num_heads, self.hidden_dim, self.mlp_dim, dropout,
                                    attention_dropout, nn.LayerNorm)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        self.heads = nn.Sequential(heads_layers)

        for conv in [self.conv_proj1, self.conv_proj2, self.conv_proj3]:
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            nn.init.trunc_normal_(conv.weight, std=math.sqrt(1 / fan_in))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape

        n_h = h // self.patch_size
        n_w = w // self.patch_size

        x = self.relu(self.bn1(self.conv_proj1(x)))
        x = self.relu(self.bn2(self.conv_proj2(x)))
        x = self.conv_proj3(x)
        assert x.shape[2] == n_h, "x.shape[3]==n_h"
        assert x.shape[3] == n_w, "x.shape[3]==n_w"

        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        x = x.permute(0, 2, 1)
        return x

    def forward(self, input: torch.Tensor):
        x = self._process_input(input)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        x = x[:, 0]

        x = self.heads(x)

        return x
