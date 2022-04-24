import os.path as osp
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as LR
import numpy as np

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

        self.conv1 = GATConv(in_channels, 4, heads=4, dropout=0)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(4 * 4, 8, heads = 8, dropout=0)
        self.conv3 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0, training=self.training)
        x = self.conv3(x, edge_index)
        return x
        # return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(dataset.num_features, dataset.num_classes).to(device)
model = Net(2, 4).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

scheduler = LR.StepLR(optimizer, step_size=10, gamma=0.5)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out[data.train_mask].float(), data.y[data.train_mask].float(), reduce=True, size_average=True)
    # print(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        # print(mask)
        x = data.y[mask] - out[mask]
        acc = torch.norm(x, p=2) ** 2 / x.shape[0] / x.shape[1]
        accs.append(acc)
        # print(data.y[mask])
        # print(out[mask])
    return accs


for epoch in range(1, 201):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')