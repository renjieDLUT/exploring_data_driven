import torch.nn as nn
import torch
import torch.nn.functional as functional


class PathTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.obj_bn1 = nn.BatchNorm1d(5)
        self.obj_fc1 = nn.Linear(5, 16)
        self.obj_fc2 = nn.Linear(16, 64)
        self.obj_fc3 = nn.Linear(64, 128)

        self.transformer = nn.Transformer(d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                                          dim_feedforward=256, batch_first=True, norm_first=True)

        # self.ref_line_bn1 = nn.BatchNorm1d(5)
        self.ref_line_fc1 = nn.Linear(2, 16)
        self.ref_line_fc2 = nn.Linear(16, 64)
        self.ref_line_fc3 = nn.Linear(64, 128)

        self.pred_fc1 = nn.Linear(128, 16)
        self.pred_fc2 = nn.Linear(16, 3)

    def forward(self, obj, ref_line):
        obj_embedding = functional.relu(self.obj_fc3(functional.relu(self.obj_fc2(functional.relu(self.obj_fc1(obj))))))

        ref_line_embedding = functional.relu(
            self.ref_line_fc3(functional.relu(self.ref_line_fc2(functional.relu(self.ref_line_fc1(ref_line))))))

        transformer_out = self.transformer(obj_embedding, ref_line_embedding)

        pred = self.pred_fc2(functional.relu(self.pred_fc1(transformer_out)))
        return pred
