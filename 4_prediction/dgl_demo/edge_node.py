import dgl
import torch

g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
print(g)
print(g.nodes())
print(g.edges())
g.ndata['x'] = torch.ones(g.num_nodes(), 3)
g.edata['x'] = torch.ones(g.num_edges(), dtype=torch.int32)
g.ndata['y'] = torch.randn(g.num_nodes(), 5)