import torch
import numpy as np

# ================== initializing a Tensor =====================
# ------------------  directly from data -----------------------
data = [[1, 2], [3, 4]]
x = torch.tensor(data)

# ------------------  from a numpy data  -----------------------
np_array = np.array(data)
x = torch.from_numpy(np_array)
x = torch.tensor(np_array)

# ------------------  from another tensor  -----------------------
x_ones = torch.ones_like(x)
x_rand = torch.rand_like(x, dtype=torch.float32)
x_randint = torch.randint_like(x, 10)

# ------------------  with random or constant values  -----------------------
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# ================== attributes of a tensor ==============================
data = torch.rand(3, 4)
print(f'shape of tensor:{data.shape}')
print(f'datatype of tensor:{data.dtype}')
print(f'device of tensor:{data.device}')


# ================== operations on tensor ==============================
# ------------------  indexing and slicing -----------------------------
print(f"first row:{data[0]}")
print(f"first column:{data[:,0]}")
print(f"last column:{data[:,-1]}")
data[:, 1] = 0
print(data)

# ------------------  joining tensor -----------------------------
concat_data = torch.cat([data, data, data], dim=1)

# ------------------ arithmetic operations -----------------------
data = torch.ones(2, 3)
y1 = data@data.t()
y2 = torch.matmul(data, data.t())

y3 = torch.rand_like(y1)
torch.matmul(data, data.t(), out=y3)

# ----------------- element-wise product -------------------------
y1 = data*data
y2 = torch.mul(data, data)
y3 = torch.ones_like(y1)
torch.mul(data, data, out=y3)

# ------------------ single-element tensor -------------------------
agg = data.sum()
agg_item = agg.item()
print(f'agg_item:{agg_item}')

# ------------------ in-place tensor -------------------------
# 以 _ 为后缀。in-place 操作节省memory,但在计算导数时可能会出现问题，因为历史记录会立即丢失
data.add_(1)


x = torch.full((3,), 10)
w = torch.ones(3, requires_grad=True)
b = torch.ones(3, requires_grad=True)

y = x * w + b
print(y.shape, w.grad)
y1 = y[:2]
y1 = y1.sum()
y1.backward()
print(w.grad)
