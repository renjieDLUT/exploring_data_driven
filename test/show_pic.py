import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as ft

if __name__ == "__main__":
    img = Image.open("./res/lena.png")
    print(type(img))
    img = np.asarray(img)
    print(type(img), img.shape)
    print("----------rotate image---------")
    img_tensor = torch.from_numpy(img)
    h, w, c = img.shape
    rotated_img_tensor = ft.rotate(img_tensor.permute(2, 0, 1), 60, center=(h // 2, w // 2))
    rotated_img = rotated_img_tensor.permute(1, 2, 0)
    plt.imshow(rotated_img)
    plt.show()
