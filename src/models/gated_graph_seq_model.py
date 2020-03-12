import torch
import torch.nn.functional as F


class GGSModel(torch.nn.Module):
    def __init__(self, convType, dataset, channels, dropout=0.8):
        super(GGSModel, self).__init__()
        self.dropout = dropout
        channels = [dataset.num_node_features] + channels + [dataset.num_classes]
        self.conv = []
        for i in range(1, len(channels)):
            conv = convType(channels[i])
            self.add_module(str(i), conv)
            self.conv.append(conv)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.conv[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)

        # Last layer
        x = self.conv[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x