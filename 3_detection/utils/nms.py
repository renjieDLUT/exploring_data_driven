from torchvision.ops.boxes import nms
import torch

boxes=torch.rand(1000,4)
scores=torch.rand(1000)

index1=nms(boxes=boxes,scores=scores,iou_threshold=0.5)
print(index1.shape)

def my_nms(boxes,scores,iou_threshold):
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    area=(x2-x1)*(y2-y1)
    order=torch.argsort(scores,descending=True)
    
    index=[]
    while order.numel() > 0:
        index.append(order[0])
        cur=order[0]
        xx1=torch.maximum(x1[cur],x1[order[1:]])
        yy1=torch.maximum(y1[cur],y1[order[1:]])
        xx2=torch.minimum(x2[cur],x2[order[1:]])
        yy2=torch.minimum(y2[cur],y2[order[1:]])

        w=torch.maximum(torch.tensor([0.]),xx2-xx1)
        h=torch.maximum(torch.tensor([0.]),yy2-yy1)
        inter=w*h

        iou=inter/(area[cur]+area[order[1:]]-inter)

        ind=torch.where(iou<iou_threshold)[0]
        order=order[ind+1]

    return torch.tensor(index)

index2=my_nms(boxes,scores,0.5)

print(torch.all(index1==index2))


