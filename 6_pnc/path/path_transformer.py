import torch.nn as nn
import torch
import torch.nn.functional as functional


class PathTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.obj_bn1 = nn.BatchNorm1d(5)
        self.obj_fc1 = nn.Linear(10, 16)
        self.obj_fc2 = nn.Linear(16, 64)
        self.obj_fc3 = nn.Linear(64, 64)

        # self.ref_line_bn1 = nn.BatchNorm1d(5)
        self.ref_line_fc1 = nn.Linear(2, 16)
        self.ref_line_fc2 = nn.Linear(16, 64)
        # self.ref_line_fc3 = nn.Linear(64, 128)

        self.obj_decoder_layer_1 = nn.TransformerDecoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                                              batch_first=True)
        self.ref_line_decoder_layer_1 = nn.TransformerDecoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                                                   batch_first=True)

        self.obj_decoder_layer_2 = nn.TransformerDecoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                                              batch_first=True)
        self.ref_line_decoder_layer_2 = nn.TransformerDecoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                                                   batch_first=True)

        self.loc_fc1 = nn.Linear(2, 16)
        self.loc_fc2 = nn.Linear(16, 64)

        self.pos_embedding = nn.Parameter(torch.empty(21, 64).normal_())

        self.pred_fc1 = nn.Linear(64, 16)
        self.pred_fc2 = nn.Linear(16, 2)

    def forward(self, obj, ref_line, loc):
        bs, _ = loc.shape
        obj_embedding = functional.relu(self.obj_fc3(functional.relu(self.obj_fc2(functional.relu(self.obj_fc1(obj))))))
        ref_line_embedding = functional.relu(self.ref_line_fc2(functional.relu(self.ref_line_fc1(ref_line))))

        loc_embedding = functional.relu(self.loc_fc2(functional.relu(self.loc_fc1(loc))))

        query = self.pos_embedding.unsqueeze(dim=0).repeat(bs, 1, 1) + loc_embedding.unsqueeze(dim=1).repeat(1, 21, 1)

        query = self.obj_decoder_layer_1(query, obj_embedding)
        query = self.ref_line_decoder_layer_1(self.pos_embedding + query, ref_line_embedding)

        query = self.obj_decoder_layer_2(self.pos_embedding + query, obj_embedding)
        query = self.ref_line_decoder_layer_2(self.pos_embedding + query, ref_line_embedding)

        # pred = self.pred_fc2(functional.relu(self.pred_fc1(transformer_out)))
        pred = self.pred_fc2(functional.relu(self.pred_fc1(query)))

        return pred
