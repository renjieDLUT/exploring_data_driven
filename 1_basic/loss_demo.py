import math

import torch.nn as nn
import torch

# ================== cross entropy cross ==============
# -10 * p * log(q)
# -10 * 0.1 * log(0.1)
loss_fn = nn.CrossEntropyLoss()
print("-" * 20 + loss_fn.__class__.__name__ + "-" * 20)
y_hat = torch.full((1, 10), 0.1)
y = torch.full((1, 10), 0.1)
loss = loss_fn(y_hat, y)
print(f'loss:{loss.item()}   expection:{math.log(10.)}')

# -1 * log(0.1)
y = torch.tensor([3])
loss = loss_fn(y_hat, y)
print(f'loss:{loss.item()}   expection:{math.log(10.)}')


