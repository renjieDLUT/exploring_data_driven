import torch
import torch.nn as nn
from typing import List

class RPNHead(nn.Module):
    def __init__(self, in_channel: int, num_anchors: int):
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                                    nn.BatchNorm2d(in_channel),
                                    nn.ReLU()
                                    )
        self.cls_conv=nn.Conv2d(in_channel,num_anchors*2,1,1)
        self.regression_conv=nn.Conv2d(in_channel,num_anchors*4,1,1)

    def forward(self,x:List[torch.Tensor]):
        logits=[]
        bbox_reg=[]
        for feature in x:
            t=self.conv_1(feature)
            logits.append(self.cls_conv(t))
            bbox_reg.append(self.regression_conv(t))
        return logits,bbox_reg

