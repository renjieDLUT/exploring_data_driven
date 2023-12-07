import torch.nn as nn
import torch
import torch.nn.functional as functional


class PathTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.obj_bn1 = nn.BatchNorm1d(5)
        self.obj_fc1 = nn.Linear(5, 16)
        self.obj_fc2 = nn.Linear(16, 64)
        self.obj_fc3 = nn.Linear(64, 64)

        decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 1)
        # self.transformer = nn.Transformer(d_model=64, nhead=4, num_encoder_layers=1, num_decoder_layers=1,dropout=0.0,
        #                                   dim_feedforward=128, batch_first=True, norm_first=True)
        #

        self.loc_fc1 = nn.Linear(2, 16)
        self.loc_fc2 = nn.Linear(16, 64)

        # self.ref_line_bn1 = nn.BatchNorm1d(5)
        self.ref_line_fc1 = nn.Linear(2, 16)
        self.ref_line_fc2 = nn.Linear(16, 64)
        # self.ref_line_fc3 = nn.Linear(64, 128)

        self.mlp_1 = nn.Linear(64, 64)
        self.mlp_2 = nn.Linear(64, 64)

        self.pred_fc1 = nn.Linear(64, 16)
        self.pred_fc2 = nn.Linear(16, 2)

    def forward(self, obj, ref_line, loc):
        obj_embedding = functional.relu(self.obj_fc3(functional.relu(self.obj_fc2(functional.relu(self.obj_fc1(obj))))))

        loc_embedding = functional.relu(self.loc_fc2(functional.relu(self.loc_fc1(loc))))
        ref_line_embedding = functional.relu(self.ref_line_fc2(functional.relu(self.ref_line_fc1(ref_line))))

        query = functional.relu(self.mlp_1(functional.relu(self.mlp_1(loc_embedding + ref_line_embedding))))

        transformer_out = self.decoder(query, obj_embedding)

        # pred = self.pred_fc2(functional.relu(self.pred_fc1(transformer_out)))
        pred = self.pred_fc2(functional.relu(self.pred_fc1(transformer_out)))

        return pred
