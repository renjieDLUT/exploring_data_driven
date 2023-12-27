import torch
from torch_geometric.data import Data

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

# 验证图的有效性,当edge_index中出现大于节点个数的值时报错
data.validate(raise_on_error=True)
print(data.keys)
print(data['x'])
print(data.num_nodes)
print(data.num_edges)
print(data.num_node_features)
print(data.has_isolated_nodes())
print(f'Has_self_loops:{data.has_self_loops()}')
print(f'Is_directed:{data.is_directed()}')
device = torch.device('cuda')
data = data.to(device)
