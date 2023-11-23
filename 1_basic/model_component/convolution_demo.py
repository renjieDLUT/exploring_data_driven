import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

img_path = "./res/lena.png"
img = Image.open(img_path)
img_np: np.ndarray = np.asarray(img).astype(np.float32)
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(dim=0)

# ref.to https://zhuanlan.zhihu.com/p/595972115
fig = plt.figure()
for i in range(4):
    conv = nn.Conv2d(3, 3, 3, 1, 1)
    prewitt_op = torch.zeros(3, 3)
    if i == 0:
        prewitt_op[:, 0] = -1
        prewitt_op[:, 2] = 1
    elif i == 1:
        prewitt_op[0, :] = -1
        prewitt_op[2, :] = 1
    elif i == 3:
        prewitt_op[:, 0] = torch.tensor([-1, -2, -1])
        prewitt_op[:, 2] = torch.tensor([1, 2, 1])
    elif i == 4:
        prewitt_op[0, :] = torch.tensor([-1, -2, -1])
        prewitt_op[2, :] = torch.tensor([1, 2, 1])
    prewitt_op = torch.stack([prewitt_op]*3)
    prewitt_op = torch.stack([prewitt_op]*3)
    weight = nn.parameter.Parameter(prewitt_op)
    conv.weight.data = weight
    img_conv = conv(img_tensor).squeeze().permute(1, 2, 0).to(torch.uint8)
    ax = fig.add_subplot(2, 2, i+1)
    ax.imshow(img_conv.data)
plt.show()

conv = torch.nn.Conv2d(3, 3, 3, 1, 1)
conv.weight.data = torch.full((3, 3, 3, 3), 1/27)
img_conv = conv(img_tensor).squeeze().permute(1, 2, 0).to(torch.uint8)
plt.imshow(img_conv.data)
plt.show()

# (C, H, W) -> (C*K*K, H*W)


def img2col(img_tensor: torch.Tensor, kernal_size=3, stride=1):
    channel, h, w = img_tensor.shape
    padding_tensor = torch.zeros(channel, h+2, w+2)
    padding_tensor[:, 1:h+1, 1:w+1] = img_tensor
    res_tensor = torch.zeros(channel*kernal_size*kernal_size, h*w)
    for i in range(1, h+1):
        for j in range(1, w+1):
            index = (i-1)*w+j-1
            tmp = torch.zeros(channel*kernal_size*kernal_size)
            for c in range(channel):
                for k in range(3):
                    for m in range(3):
                        tmp[c*9+k*3+m] = padding_tensor[c, i-1+k, j-1+m]
            res_tensor[:, index] = tmp
    return res_tensor

# (O, H*W) -> (O, H, W)


def col2img(matrix_tensor: torch.Tensor, out_channel, h, w, k=3):
    assert matrix_tensor.dim() == 2
    r = matrix_tensor.shape[0]
    assert matrix_tensor.shape[0] == out_channel
    assert matrix_tensor.shape[1] == h*w
    feature_map = matrix_tensor.reshape(r, h, w)
    return feature_map

# (O, C, H, W) -> (O, C*H*W)


def trans_conv2d(conv: torch.Tensor):
    print(type(conv), conv.shape, conv.dim)
    assert conv.dim() == 4
    o, c, k1, k2 = conv.shape
    return conv.reshape(o, c*k1*k2)


t1 = img2col(img_tensor.squeeze())
my_conv = torch.full((3, 3, 3, 3), 1/27)
t2 = trans_conv2d(my_conv)
t3 = t2@t1
out = col2img(t3, 3, img_tensor.shape[-2], img_tensor.shape[-1])
out_img = out.permute(1, 2, 0).to(torch.uint8)
plt.imshow(out_img)
plt.show()
