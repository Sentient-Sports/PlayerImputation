"""
Graph Neural Network module of the model.
"""


import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import Linear, Parameter

class GCN(torch.nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.conv1 = SAGEConv(input_size, 64)
        self.conv2 = SAGEConv(64, 32)
        self.linear2 = Linear(32,2) 
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        return x
