import torch
import torch.nn as nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=nheads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, batch_first=True)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.pos_query = nn.Embedding()
        self.obj_query = nn.Parameter(torch.randn(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.randm(50, hidden_dim // 2))
        self.col_embed == nn.Parameter(torch.randn(50, hidden_dim // 2))

    def forward(self, inputs):
        bs, _, _, _ = inputs.shape
        feature = self.backbone(inputs)
        H, W = feature.shape[-2:]
        embedding = self.conv(feature)
        query = embedding.flatten(2).permute(0, 2, 1)
        pos_embedding = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(0)
        transformer_out = self.transformer(pos_embedding + query, self.obj_query.repeat(bs, 1, 1))
        return self.linear_class(transformer_out),self.linear_bbox(transformer_out).sigmoid()

