import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_geometric.nn import APPNP

class MonoAPPNPModel(torch.nn.Module):
    def __init__(self, dataset, channels, dropout=0.8, K=10, alpha=0.10):
        super(MonoAPPNPModel, self).__init__()
        self.dropout = dropout

        if len(channels) > 1:
            print('WARNING: Taking only the first hidden layer size, the rest is ignored.')

        self.nn = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dataset.num_node_features, channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[0], dataset.num_classes),
        )
        self.appnp = APPNP(K,alpha)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.nn(x)
        x = self.appnp(x,edge_index)
        x = F.log_softmax(x, dim=1)

        return x