import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.anchor_utils import AnchorGenerator

from voc_dataset import VOCDataset,collate_fn

dataset = VOCDataset()
dataloader = DataLoader(dataset, 2,collate_fn=collate_fn)

x, y = dataset[0]
print(x.shape, y)

backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).features
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

faster_rcnn = torchvision.models.detection.FasterRCNN(backbone, num_classes=21, rpn_anchor_generator=anchor_generator,
                                                      box_roi_pool=roi_pooler)
# faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
#     weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
#     weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

optimizer=torch.optim.SGD(faster_rcnn.parameters(),lr=0.01)


epochs =5

faster_rcnn.train()
for epoch in range(epochs):
    for x,y in dataloader:
        print(type(x),y)
        loss=faster_rcnn(x,y)
        for l in loss:
            l.backward()
        optimizer.step()
        optimizer.zero_grad()
