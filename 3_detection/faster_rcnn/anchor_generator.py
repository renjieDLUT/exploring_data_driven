import torch.nn as nn
from typing import List
import torch

from torchvision.models.detection.image_list import ImageList
import torchvision


class AnchorGenerator(nn.Module):
    def __init__(self, sizes=((128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [self.generate_anchors(scale, aspect_ratio) for scale, aspect_ratio in
                             zip(sizes, aspect_ratios)]

    def generate_anchors(self, scales: List[int], aspect_ratios: List[float]) -> torch.Tensor:
        scales = torch.tensor(scales)
        aspect_ratios = torch.tensor(aspect_ratios)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratio = 1 / h_ratios

        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        ws = (w_ratio[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2  # (9, 4)
        return base_anchors.round()

    def get_anchors(self, grid_sizes: List[List[int]], strides: List[List[torch.Tensor]]):
        anchors = []
        cell_anchors = self.cell_anchors

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)
            shifts_x = shifts_x.reshape(-1)
            shifts_y = shifts_y.reshape(-1)
            shift = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=1)  # eg: feature_map(10, 10) -> (100, 4)
            anchors.append((shift[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4))
        anchors = torch.cat(anchors, dim=0)
        return anchors

    def forward(self, image_list: ImageList, feature_maps=List[torch.Tensor]):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
            ]
            for g in grid_sizes
        ]
        anchors_over_all_feature_maps = self.get_anchors(grid_sizes, strides)
        anchors = []
        anchors.append(anchors_over_all_feature_maps)
        anchors = anchors * image_list.tensors.shape[0]
        return anchors


if __name__ == "__main__":
    anchor_generator = AnchorGenerator()
    fake_imgs = torch.randn((8, 3, 224, 224))
    resnet = torchvision.models.resnet18()
    features_net = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                                 resnet.layer3, resnet.layer4)
    feature = features_net(fake_imgs)
    feature_maps = []
    print(feature.shape)
    feature_maps.append(feature)
    image_sizes = [(fake_imgs.shape[-2], fake_imgs.shape[-1])] * fake_imgs.shape[0]
    image_list = ImageList(fake_imgs, image_sizes)
    anchors = anchor_generator(image_list, feature_maps)
    print(len(anchors), anchors[0].shape)  # (441, 4)
