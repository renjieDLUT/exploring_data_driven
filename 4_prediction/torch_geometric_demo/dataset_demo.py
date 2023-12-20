from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

dataset = TUDataset('./data/', name='ENZYMES')
print(len(dataset))
print(dataset[0])
print(dataset.num_classes)
print(dataset.num_node_features)
dataset = Planetoid(root='./data/', name='Cora')
data=dataset[0]

loader=DataLoader(dataset,batch_size=32,shuffle=True)
batch=next(iter(loader))
print(batch)
x=scatter(batch.x, batch.batch, dim=0, reduce='mean')
print(x.shape)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    print(out.shape,data.y.shape)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
