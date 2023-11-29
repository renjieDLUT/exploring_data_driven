import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
path = "/home/renjie/renjie_ws/dataset/PASCAL VOC"

train_transforms = transforms.Compose([transforms.ToTensor()])

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

class_to_index = {cls: i for i, cls in enumerate(VOC_BBOX_LABEL_NAMES)}


class target_transforms:

    def __call__(self, target):
        boxes = []
        label = []
        object = target['annotation']['object']
        for obj in object:
            bndbox = obj['bndbox']
            boxes.append([float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(bndbox['ymax'])])
            label.append(class_to_index[obj['name']])
        res = {}
        res['boxes'] = torch.tensor(boxes).to(device="cuda")
        res['labels'] = torch.tensor(label).to(dtype=torch.int64).to(device="cuda")
        return res

def collate_fn(data):
    imgs=[]
    label=[]
    for sample in data:
        imgs.append(sample[0].to(device="cuda"))
        label.append(sample[1])
    return imgs,label

class VOCDataset(Dataset):
    def __init__(self):
        self.voc_detection = torchvision.datasets.VOCDetection(path, transform=train_transforms,target_transform=target_transforms())

    def __len__(self):
        return len(self.voc_detection)

    def __getitem__(self, item):
        return self.voc_detection[item]


if __name__=="__main__":
    dataset=VOCDataset()
    x,y=dataset[0]
    print(x.shape,y)

