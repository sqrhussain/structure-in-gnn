
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np

class MonoGAT(torch.nn.Module):
    def __init__(self,dataset,channels,heads=1,dropout=0.6,attention_dropout=0.3):
        super(MonoGAT,self).__init__()
        channels = [dataset.num_node_features] + channels + [dataset.num_classes]
        
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        self.conv = []
        for i in range(1,len(channels)):
            if i == 1:
                conv = GATConv(channels[i-1],channels[i],heads=heads,dropout=self.attention_dropout)
            elif i == len(channels)-1:
                if dataset.name=='PubMed':
                    conv = GATConv(channels[i-1]*heads,channels[i],heads=heads,concat=False,dropout=self.attention_dropout)
                else:
                    conv = GATConv(channels[i-1]*heads,channels[i],dropout=self.attention_dropout)
            else:
                conv = GATConv(channels[i-1]*heads,channels[i],heads=heads,dropout=self.attention_dropout)
            self.add_module(str(i),conv)
            self.conv.append(conv)
        
        
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index 
        
        for conv in self.conv[:-1]:
            x = conv(x,edge_index)
            x = F.elu(x)
            x = F.dropout(x,p=self.dropout,training=self.training) # YOU MUST UNDERSTAND DROPOUT (in attention)
        
        # Last layer
        x = self.conv[-1](x,edge_index)
        x = F.softmax(x,dim=1)
        
        return x
    

class BiGAT(torch.nn.Module):
    def __init__(self,dataset,channels,heads=1,dropout=0.6,attention_dropout=0.3):
        super(BiGAT,self).__init__()
        self.conv_st = []
        self.conv_ts = []
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        channels_output = [dataset.num_node_features] + [c*2*heads for c in channels]
        channels = [dataset.num_node_features] + channels
        for i in range(len(channels)-1):
            conv_st = GATConv(channels_output[i], channels[i+1],heads=heads,dropout=self.attention_dropout)
            self.add_module('conv_st'+str(i),conv_st)
            self.conv_st.append(conv_st)
            
            conv_ts = GATConv(channels_output[i], channels[i+1],heads=heads,dropout=self.attention_dropout)
            self.add_module('conv_ts'+str(i),conv_ts)
            self.conv_ts.append(conv_ts)
        
        if dataset.name=='PubMed':
            self.last = GATConv(channels_output[-1], dataset.num_classes,heads=heads,concat=False,dropout=self.attention_dropout)
        else:
            self.last = GATConv(channels_output[-1], dataset.num_classes,dropout=self.attention_dropout)
        
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index
        st_edges = data.edge_index.t()[1-data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
#         print(ts_edges.shape)
        for i in range(len(self.conv_st)):
            x1 = F.elu(self.conv_st[i](x,st_edges))
            x2 = F.elu(self.conv_ts[i](x,ts_edges))
            x = torch.cat((x1,x2),dim=1)
            x = F.dropout(x,p=self.dropout,training=self.training)
        
        # last layer
        x = self.last(x,edge_index)
        x = F.softmax(x,dim=1) 
        
        return x
    
class TriGAT(torch.nn.Module):
    def __init__(self,dataset,channels,heads=1,dropout=0.6,attention_dropout=0.3):
        super(TriGAT,self).__init__()
        self.conv_st = []
        self.conv_ts = []
        self.conv = []
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        channels_output = [dataset.num_node_features] + [c*3*heads for c in channels]
        channels = [dataset.num_node_features] + channels
        for i in range(len(channels)-1):
            conv_st = GATConv(channels_output[i], channels[i+1],heads=heads,dropout=self.attention_dropout)
            self.add_module('conv_st'+str(i),conv_st)
            self.conv_st.append(conv_st)
            
            conv_ts = GATConv(channels_output[i], channels[i+1],heads=heads,dropout=self.attention_dropout)
            self.add_module('conv_ts'+str(i),conv_ts)
            self.conv_ts.append(conv_ts)
            
            conv = GATConv(channels_output[i],channels[i+1],heads=heads,dropout=self.attention_dropout)
            self.add_module('conv'+str(i),conv)
            self.conv.append(conv)
        
        if dataset.name=='PubMed':
            self.last = GATConv(channels_output[-1], dataset.num_classes,heads=heads,concat=False,dropout=self.attention_dropout)
        else:
            self.last = GATConv(channels_output[-1], dataset.num_classes,dropout=self.attention_dropout)
        
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index
        st_edges = data.edge_index.t()[1-data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
#         print(ts_edges.shape)
        for i in range(len(self.conv_st)):
            x1 = F.elu(self.conv_st[i](x,st_edges))
            x2 = F.elu(self.conv_ts[i](x,ts_edges))
            x3 = F.elu(self.conv[i](x,edge_index))
            x = torch.cat((x1,x2,x3),dim=1)
            x = F.dropout(x,p=self.dropout,training=self.training)
        
        # last layer
        x = self.last(x,edge_index)
        x = F.softmax(x,dim=1) 
        
        return x