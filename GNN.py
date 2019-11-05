import torch
import torch.nn.functional as F
import numpy as np


class MonoModel(torch.nn.Module):
    def __init__(self, convType, dataset, channels):
        super(MonoModel, self).__init__()
        channels = [dataset.num_node_features] + channels + [dataset.num_classes]
        self.conv = []
        for i in range(1, len(channels)):
            conv = convType(channels[i - 1], channels[i])
            self.add_module(str(i), conv)
            self.conv.append(conv)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.conv[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        # Last layer
        x = self.conv[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


class BiModel(torch.nn.Module):
    def __init__(self, convType, dataset, channels):
        super(BiModel, self).__init__()
        self.conv_st = []
        self.conv_ts = []
        channels_output = [dataset.num_node_features] + [c * 2 for c in channels]
        channels = [dataset.num_node_features] + channels
        for i in range(len(channels) - 1):
            conv_st = convType(channels_output[i], channels[i + 1])
            self.add_module('conv_st' + str(i), conv_st)
            self.conv_st.append(conv_st)

            conv_ts = convType(channels_output[i], channels[i + 1])
            self.add_module('conv_ts' + str(i), conv_ts)
            self.conv_ts.append(conv_ts)

        self.last = convType(channels_output[-1], dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        st_edges = data.edge_index.t()[1 - data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
        #         print(ts_edges.shape)
        for i in range(len(self.conv_st)):
            x1 = F.relu(self.conv_st[i](x, st_edges))
            x2 = F.relu(self.conv_ts[i](x, ts_edges))
            x = torch.cat((x1, x2), dim=1)
            x = F.dropout(x, training=self.training)

        # last layer
        x = self.last(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


class TriModel(torch.nn.Module):
    def __init__(self, convType, dataset, channels):
        super(TriModel, self).__init__()
        self.conv_st = []
        self.conv_ts = []
        self.conv = []
        channels_output = [dataset.num_node_features] + [c * 3 for c in channels]
        channels = [dataset.num_node_features] + channels
        for i in range(len(channels) - 1):
            conv_st = convType(channels_output[i], channels[i + 1])
            self.add_module('conv_st' + str(i), conv_st)
            self.conv_st.append(conv_st)

            conv_ts = convType(channels_output[i], channels[i + 1])
            self.add_module('conv_ts' + str(i), conv_ts)
            self.conv_ts.append(conv_ts)

            conv = convType(channels_output[i], channels[i + 1])
            self.add_module('conv' + str(i), conv)
            self.conv.append(conv)

        self.last = convType(channels_output[-1], dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        st_edges = data.edge_index.t()[1 - data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
        #         print(ts_edges.shape)
        for i in range(len(self.conv_st)):
            x1 = F.relu(self.conv_st[i](x, st_edges))
            x2 = F.relu(self.conv_ts[i](x, ts_edges))
            x3 = F.relu(self.conv[i](x, edge_index))
            x = torch.cat((x1, x2, x3), dim=1)
            x = F.dropout(x, training=self.training)

        # last layer
        x = self.last(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x