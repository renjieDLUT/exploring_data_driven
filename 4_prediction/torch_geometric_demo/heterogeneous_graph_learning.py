from torch_geometric.datasets import OGB_MAG
from torch_geometric.utils import add_self_loops, degree

dataset=OGB_MAG('./data/ogb_mag')
data=dataset[0]
print(data)
print(data['author'])
print(data.has_self_loops())
homogeneous_data = data.to_homogeneous()
print(homogeneous_data)
import torch_geometric.transforms as T
data = T.AddSelfLoops()(data)
print(data.metadata())
print(dataset.num_classes)
