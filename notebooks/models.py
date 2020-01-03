
import torch
import torch.nn.functional as F

class MonoModel(torch.nn.Module):
    def __init__(self,conv,dataset,channels1):
        super(MonoModel,self).__init__()
        self.conv1 = conv(dataset.num_node_features, channels1) 
        self.conv2 = conv(channels1,dataset.num_classes)
        
    def forward(self, data): 
        x, edge_index = data.x, data.edge_index 
        # First layer
        h1_ = self.conv1(x,edge_index)
        h1 = F.relu(h1_)
        h1 = F.dropout(h1,training=self.training) # YOU MUST UNDERSTAND DROPOUT
        
        # Second layer
        h2_ = self.conv2(h1,edge_index)
        h2 = F.log_softmax(h2_,dim=1)
        
        return h2
    

class BiModel(torch.nn.Module):
    def __init__(self,conv,dataset,channels_st,channels_ts):
        super(BiModel,self).__init__()
        self.conv_st1 = conv(dataset.num_node_features, channels_st) 
        self.conv_st2 = conv(channels_st+channels_ts,dataset.num_classes)
        
        self.conv_ts1 = conv(dataset.num_node_features, channels_ts) 
        self.conv_ts2 = conv(channels_st+channels_ts,dataset.num_classes)
        
        
        self.conv_2 = conv(channels_st+channels_ts,dataset.num_classes)
        
        self.linear = torch.nn.Linear(2,1)
        
    def forward(self, data): 
        x = data.x  
        st_edges = data.edge_index.t()[1-data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
        # First layer
        st_h1 = F.relu(self.conv_st1(x,st_edges))
        ts_h1 = F.relu(self.conv_ts1(x,ts_edges))
        h1 = torch.cat((st_h1,ts_h1),dim=1)
        h1 = F.dropout(h1,training=self.training) # ===== YOU MUST UNDERSTAND DROPOUT =====
        
        # Second layer
#         st_h2 = self.conv_st2(h1,st_edges)
#         ts_h2 = self.conv_ts2(h1,ts_edges)
#         h2 = torch.stack([st_h2,ts_h2],dim=2)
#         h2 = self.linear(h2)[:,:,0]
        h2 = self.conv_2(h1,data.edge_index)
        h2 = F.log_softmax(h2,dim=1) 
        
        return h2
    
class TriModel(torch.nn.Module):
    def __init__(self,conv,dataset,channels,channels_st,channels_ts):
        super(TriModel,self).__init__()
        self.conv_st1 = conv(dataset.num_node_features, channels_st) 
        self.conv_st2 = conv(channels_st+channels_ts+channels,dataset.num_classes)
        self.conv_1 = conv(dataset.num_node_features,channels)
        
        self.conv_ts1 = conv(dataset.num_node_features, channels_ts) 
        self.conv_ts2 = conv(channels_st+channels_ts+channels,dataset.num_classes)
        self.conv_2 = conv(channels_st+channels_ts+channels,dataset.num_classes)
        
        self.linear = torch.nn.Linear(3,1)
        
    def forward(self, data): 
        x = data.x
        edge_index = data.edge_index
        st_edges = edge_index.t()[1-data.is_reversed].t()
        ts_edges = edge_index.t()[data.is_reversed].t()
        # First layer
        st_h1 = F.relu(self.conv_st1(x,st_edges))
        ts_h1 = F.relu(self.conv_ts1(x,ts_edges))
        _h1 = F.relu(self.conv_1(x,edge_index))
        h1 = torch.cat((st_h1,ts_h1,_h1),dim=1)
        h1 = F.dropout(h1,training=self.training) # ===== YOU MUST UNDERSTAND DROPOUT =====
        
        # Second layer
#         st_h2 = self.conv_st2(h1,st_edges)
#         ts_h2 = self.conv_ts2(h1,ts_edges)
#         _h2 = self.conv_2(h1,edge_index)
#         h2 = torch.stack([st_h2,ts_h2,_h2],dim=2)
#         h2 = self.linear(h2)[:,:,0]
        h2 = self.conv_2(h1,data.edge_index)
        h2 = F.log_softmax(h2,dim=1) 
        
        return h2
    
class TriPreModel(torch.nn.Module):
    def __init__(self,conv,dataset,channels,channels_st,channels_ts,chennel_intemediate):
        super(TriPreModel,self).__init__()
        self.conv_st1 = conv(dataset.num_node_features, channels_st)
        self.conv_ts1 = conv(dataset.num_node_features, channels_ts)
        self.conv_1 = conv(dataset.num_node_features,channels)
        
        self.conv_2 = conv(channels_st+channels_ts+channels,chennel_intemediate)
        self.conv_3 = conv(chennel_intemediate,dataset.num_classes)
        
        self.linear = torch.nn.Linear(3,1)
        
    def forward(self, data): 
        x = data.x
        edge_index = data.edge_index
        st_edges = edge_index.t()[1-data.is_reversed].t()
        ts_edges = edge_index.t()[data.is_reversed].t()
        # First layer
        st_h1 = F.relu(self.conv_st1(x,st_edges))
        ts_h1 = F.relu(self.conv_ts1(x,ts_edges))
        _h1 = F.relu(self.conv_1(x,edge_index))
        h1 = torch.cat((st_h1,ts_h1,_h1),dim=1)
        h1 = F.dropout(h1,training=self.training) 
        
        # Second layer
        h2 = self.conv_2(h1,data.edge_index)
        h2 = F.dropout(h2,training=self.training)
        # Third layer
        
        h3 = self.conv_3(h2,data.edge_index)
        h3 = F.log_softmax(h3,dim=1) 
        
        return h3
    
class TriLateModel(torch.nn.Module):
    def __init__(self,conv,dataset,channels,channels_st,channels_ts,chennel_intemediate):
        super(TriLateModel,self).__init__()
        self.conv_1 = conv(dataset.num_node_features,chennel_intemediate)
        
        self.conv_2 = conv(chennel_intemediate,channels)
        self.conv_st2 = conv(chennel_intemediate, channels_st)
        self.conv_ts2 = conv(chennel_intemediate, channels_ts)
        
        self.conv_3 = conv(channels+channels_st+channels_ts,dataset.num_classes)
        
        self.linear = torch.nn.Linear(3,1)
        
    def forward(self, data): 
        x = data.x
        edge_index = data.edge_index
        st_edges = edge_index.t()[1-data.is_reversed].t()
        ts_edges = edge_index.t()[data.is_reversed].t()
        
        
        # First layer
        h1 = self.conv_1(x,data.edge_index)
        h1 = F.dropout(h1,training=self.training)
        
        # Second layer
        st_h2 = F.relu(self.conv_st2(h1,st_edges))
        ts_h2 = F.relu(self.conv_ts2(h1,ts_edges))
        _h2 = F.relu(self.conv_2(h1,edge_index))
        h2 = torch.cat((st_h2,ts_h2,_h2),dim=1)
        h2 = F.dropout(h2,training=self.training) 
        
        # Third layer
        h3 = self.conv_3(h2,data.edge_index)
        h3 = F.log_softmax(h3,dim=1) 
        
        return h3
    
    