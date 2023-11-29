from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).features
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

# faster_rcnn = torchvision.models.detection.FasterRCNN(backbone, num_classes=21, rpn_anchor_generator=anchor_generator,
#                                                       box_roi_pool=roi_pooler)
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
pic_path="./res/lena.png"
import PIL.Image as Image

categories=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]

raw_img=Image.open(pic_path)
img:torch.Tensor=transforms.ToTensor()(raw_img)
print(f"img:{img.shape}")
imgs=[img]
faster_rcnn.eval()
res=faster_rcnn(imgs)
labels=res[0]["labels"]
scores=res[0]['scores']
boxes=res[0]["boxes"]
print(res)
fig:plt.Figure=plt.figure()
ax=fig.add_subplot()
ax.imshow(raw_img)
for i in range(labels.shape[0]):
    label=labels[i].item()
    score=scores[i].item()
    print(categories[label],score)
    box=boxes[i]
    x1=box[0].item()
    y1=box[1].item()
    x2=box[2].item()
    y2=box[3].item()
    if score>0.5:
        rec=plt.Rectangle((x1,y1),(x2-x1),y2-y1, fill=False)
        ax.add_patch(rec)
plt.show()
