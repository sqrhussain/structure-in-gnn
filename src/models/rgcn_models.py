import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class MonoRGCN(torch.nn.Module):
    def __init__(self, convType, dataset, channels, dropout=0.8, num_bases=10):
        super(MonoRGCN, self).__init__()
        self.dropout = dropout
        channels = [dataset.num_node_features] + channels + [dataset.num_classes]
        self.conv = []
        for i in range(1, len(channels)):
            if convType == RGCNConv:
                conv = convType(channels[i - 1], channels[i], num_relations=2, num_bases=num_bases)
            else:
                conv = convType(channels[i - 1], channels[i], num_relations=2, num_bases=num_bases,
                            last_layer=(i + 1) == len(channels))
            self.add_module(str(i), conv)
            self.conv.append(conv)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.is_reversed.long()

        for conv in self.conv[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)

        # Last layer
        x = self.conv[-1](x, edge_index, edge_type)
        x = F.log_softmax(x, dim=1)

        return x


from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform


class RGCN2(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases, last_layer=False,
                 root_weight=True, bias=True, **kwargs):
        super(RGCN2, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.last_layer = last_layer

        if last_layer:
            self.w_size = out_channels
        else:
            self.w_size = out_channels // num_relations

        self.basis = Param(torch.Tensor(num_bases, in_channels, self.w_size))
        self.att = Param(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size will be
                automatically inferred and assumed to be quadratic.
                (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        mp_type = self.__get_mp_type__(edge_index)
        if size is None:
            size = [None, None]
        elif isinstance(size, int):
            size = [size, size]
        elif torch.is_tensor(size):
            size = size.tolist()
        elif isinstance(size, tuple):
            size = list(size)
        assert isinstance(size, list)
        assert len(size) == 2

        kwargs = self.__collect__(edge_index, size, mp_type, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        out = self.message(**msg_kwargs)

        if self.last_layer:
            aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
            out = self.aggregate(out, **aggr_kwargs)
        else:
            idx1 = kwargs['edge_type'].bool()
            idx2 = (1 - kwargs['edge_type']).bool()
            out1 = self.aggregate(out[idx1, :], kwargs['index'][idx1], None, kwargs['dim_size'])
            out2 = self.aggregate(out[idx2, :], kwargs['index'][idx2], None, kwargs['dim_size'])
            # print(out1.shape)
            # print(out2.shape)
            out = torch.cat([out1, out2], 1)
            # print(out.shape)
            # exit(0)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update(out, **update_kwargs)

        return out

    # def aggregate(self, inputs, index, dim_size):  # pragma: no cover
    #     r"""Aggregates messages from neighbors as
    #     :math:`\square_{j \in \mathcal{N}(i)}`.

    #     By default, delegates call to scatter functions that support
    #     "add", "mean" and "max" operations specified in :meth:`__init__` by
    #     the :obj:`aggr` argument.
    #     """

    #     return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.w_size)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.w_size)
            w = torch.index_select(w, 0, edge_type)
            # print(w.shape)
            # print(x_j.shape)
            # print(x_j.unsqueeze(1).shape)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
            # print(torch.bmm(x_j.unsqueeze(1), w).shape)
            # print(out.shape)
            # exit(0)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        # print(aggr_out.shape)
        # print(x.shape)
        if self.root is not None:
            if x is None:
                aggr_out = aggr_out + self.root
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)
# class BiModel(torch.nn.Module):
#     def __init__(self,convType,dataset,channels,dropout=0.8):
#         super(BiModel,self).__init__()
#         self.dropout=dropout
#         self.conv_st = []
#         self.conv_ts = []
#         channels_output = [dataset.num_node_features] + [c*2 for c in channels]
#         channels = [dataset.num_node_features] + channels
#         for i in range(len(channels)-1):
#             conv_st = convType(channels_output[i], channels[i+1])
#             self.add_module('conv_st'+str(i),conv_st)
#             self.conv_st.append(conv_st)

#             conv_ts = convType(channels_output[i], channels[i+1])
#             self.add_module('conv_ts'+str(i),conv_ts)
#             self.conv_ts.append(conv_ts)

#         self.last = convType(channels_output[-1], dataset.num_classes)

#     def forward(self, data): 
#         x, edge_index = data.x, data.edge_index
#         st_edges = data.edge_index.t()[1-data.is_reversed].t()
#         ts_edges = data.edge_index.t()[data.is_reversed].t()
# #         print(ts_edges.shape)
#         for i in range(len(self.conv_st)):
#             x1 = F.relu(self.conv_st[i](x,st_edges))
#             x2 = F.relu(self.conv_ts[i](x,ts_edges))
#             x = torch.cat((x1,x2),dim=1)
#             x = F.dropout(x,training=self.training,p=self.dropout)

#         # last layer
#         x = self.last(x,edge_index)
#         x = F.log_softmax(x,dim=1) 

#         return x

# class TriModel(torch.nn.Module):
#     def __init__(self,convType,dataset,channels,dropout=0.8):
#         super(TriModel,self).__init__()
#         self.dropout=dropout
#         self.conv_st = []
#         self.conv_ts = []
#         self.conv = []
#         channels_output = [dataset.num_node_features] + [c*3 for c in channels]
#         channels = [dataset.num_node_features] + channels
#         for i in range(len(channels)-1):
#             conv_st = convType(channels_output[i], channels[i+1])
#             self.add_module('conv_st'+str(i),conv_st)
#             self.conv_st.append(conv_st)

#             conv_ts = convType(channels_output[i], channels[i+1])
#             self.add_module('conv_ts'+str(i),conv_ts)
#             self.conv_ts.append(conv_ts)

#             conv = convType(channels_output[i],channels[i+1])
#             self.add_module('conv'+str(i),conv)
#             self.conv.append(conv)

#         self.last = convType(channels_output[-1], dataset.num_classes)

#     def forward(self, data): 
#         x, edge_index = data.x, data.edge_index
#         st_edges = data.edge_index.t()[1-data.is_reversed].t()
#         ts_edges = data.edge_index.t()[data.is_reversed].t()
# #         print(ts_edges.shape)
#         for i in range(len(self.conv_st)):
#             x1 = F.relu(self.conv_st[i](x,st_edges))
#             x2 = F.relu(self.conv_ts[i](x,ts_edges))
#             x3 = F.relu(self.conv[i](x,edge_index))
#             x = torch.cat((x1,x2,x3),dim=1)
#             x = F.dropout(x,training=self.training,p=self.dropout)

#         # last layer
#         x = self.last(x,edge_index)
#         x = F.log_softmax(x,dim=1) 

#         return x
