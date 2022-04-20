import os.path as osp
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

from torch_geometric.data import Data
from clustering import Clustering

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]
# print(data.y)


x, edge_index, y, train_mask, val_mask, test_mask = Clustering('./data/network/raw').cluster()

data = Data(x = x, edge_index = edge_index, y = y, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(dataset.num_features, dataset.num_classes).to(device)
model = Net(2, 4).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

# ###
# out = model(data.x, data.edge_index)
# print(out)
# print(data.train_mask)
# print(data.y[data.train_mask].shape)
# ###

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out[data.train_mask].float(), data.y[data.train_mask].float())
    # print(loss)
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        # print((out[mask] == data.y[mask]).sum())
        # acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        acc = float((data.y[mask] - out[mask]).sum() / mask.sum())
        accs.append(acc)
    # print(out)
    return accs


for epoch in range(1, 201):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')