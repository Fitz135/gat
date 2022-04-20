from torch_geometric.data import Data
from clustering import Clustering
import torch


x, edge_index, y, train_mask, val_mask, test_mask = Clustering('./data/network/raw').cluster()

data = Data(x = x, edge_index = edge_index, y = y, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
print(train_mask)
print(val_mask)
print(test_mask)