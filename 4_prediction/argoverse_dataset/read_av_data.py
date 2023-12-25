import pickle

import torch

file = "/home/renjie/renjie_ws/dataset/av2/test/processed/177ab5f7-cede-4c60-89ae-434c2b7f27fb.pkl"

data = None
with open(file, "rb+") as fb:
    data = pickle.load(fb)


def read_data(data: dict):
    for k, v in data.items():
        print("-" * 30)
        print(k)
        if isinstance(v, dict):
            read_data(v)
        elif isinstance(v, torch.Tensor):
            print(v.shape)
        else:
            print(v)


read_data(data)

from torch_geometric.data.hetero_data import HeteroData

heterodata=HeteroData(data)

print(heterodata)
# import pandas as pd
#
# parquet_file_path = ("/home/renjie/renjie_ws/dataset/av2/test/raw/"
#                      "177ab5f7-cede-4c60-89ae-434c2b7f27fb/scenario_177ab5f7-cede-4c60-89ae-434c2b7f27fb.parquet")
# dataframe = pd.read_parquet(parquet_file_path)
# print(dataframe)
# print(dataframe.keys())
# print(dataframe.values)
# print(dataframe.iloc[0])

