import copy

import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import time
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


def count_param(module: nn.Module):
    res = 0
    for param in module.parameters():
        res += param.numel()
    return res


def get_video_capture():
    _DEVICE_ID = 0
    _FPS = 40
    _WIDTH = 1280
    _HEIGHT = 800

    cap = cv2.VideoCapture(_DEVICE_ID)
    cap.set(cv2.CAP_PROP_FPS, _FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _HEIGHT)
    return cap, _WIDTH, _HEIGHT


def set_text():
    text_info = {}
    text_info["fontFace"] = 1
    text_info['fontScale'] = 2.
    text_info['color'] = (0, 0, 255)
    return text_info


def get_module():
    faster_rcnn_resnet50 = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    faster_rcnn_mobilenet = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
        weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    print(f'FasterRCNN_ResNet50_FPN:{count_param(faster_rcnn_resnet50):,d}   '
          f'FasterRCNN_MobileNet:{count_param(faster_rcnn_mobilenet):,d}')
    model = faster_rcnn_mobilenet
    model.to(device)
    # model = torch.quantization.quantize_dynamic(model,{nn.Conv2d}).cuda()
    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # torch.quantization.prepare(model, inplace=True)
    # torch.quantization.convert(model, inplace=True)

    model.eval()
    categories = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
    return model, categories


cap, width, height = get_video_capture()
model, categories = get_module()
text_info = set_text()
count = 0
while True:
    ret, img = cap.read()
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.to(device)
    if (count % 2) == 0:
        tic = time.time()
        res = model([img_tensor])
        print(f"cost time(inference):{time.time() - tic}")
        count = count + 1
    count += 1
    labels = res[0]["labels"]
    scores = res[0]['scores']
    boxes = res[0]["boxes"]
    text_list=[]
    for i in range(labels.shape[0]):
        label = labels[i].item()
        score = scores[i].item()
        box = boxes[i].to(dtype=torch.int32, device="cpu").data.numpy()

        if score > 0.9:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), thickness=3)
            text_info["text"] = f"{categories[label]}: {score:.3}"
            text_info['org'] = (width - box[2], box[1])
            text_list.append(copy.deepcopy(text_info))
    img = cv2.flip(img, 1)
    for text in text_list:
        cv2.putText(img, **text)
    cv2.imshow('my_detection', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
