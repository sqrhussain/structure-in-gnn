import torch
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
import pandas as pd
from random import sample, seed
import numpy as np
from torch_geometric.data import Data
import networkx as nx

def read_network(features_path,edge_path,directed,reverse):
        if not directed:
            reverse = False
        feats = []
        target = []
        rename = {}
        class_rename = {}
        cnt = 0
        class_cnt = 0
        isCoCitDataset = False
        print('Read features: RUNNING')
        with open(features_path, 'r') as f:
            for line in f:
                info = line.split()
                if len(info) == 1:
                    info = line.split(',')
                    isCoCitDataset = True
                rename[info[0]] = cnt
                feats.append(np.array([float(x) for x in info[1:-1]]))
                if info[-1] not in class_rename:
                    class_rename[info[-1]] = class_cnt
                    class_cnt+=1
                target.append(class_rename[info[-1]])
                cnt += 1
        # TF-IDF to binary BoW
        feats = (np.array(feats) > 0)

        # normalize rows (not giving good accuracy, need to investigate)
        # feats_row_sums = feats.sum(axis=1)
        # feats = feats / feats_row_sums[:, np.newaxis]
        y = torch.tensor(target,dtype=torch.long)
        n = len(target)
        
        if isCoCitDataset:
            x = torch.tensor(np.eye(n),dtype=torch.float)
        else:
            x = torch.tensor(feats,dtype=torch.float)

        print('Read features: DONE')
        # 3. Split similar to Planetoid
        
        num_classes = len(set(target))
        df = pd.DataFrame(target)
        df.columns = ['target']
        # train = []
        # seed(1)
        # for i in range(num_classes):
        #     train = train + sample(df[df['target']==i].index.values.tolist() ,20)
        # rest = [i for i in range(n) if i not in train]
        # val = rest[:500]
        # test = rest[500:1500]
        # train_ind = [1 if i in train else 0 for i in range(n)]
        # val_ind = [1 if i in val else 0 for i in range(n)]
        # test_ind = [1 if i in test else 0 for i in range(n)]
        
        print('Read edges: RUNNING')
        # 4. Read edges
#         row, col = [], []
#         edges = []
#         rev = []
#         err = 0
#         revmap = {}
#         with open(edge_path, 'r') as f:
#             for line in f:
#                 edge = line.split()
#                 if edge[0] not in rename or edge[1] not in rename:
#                     err += 1
#                     continue
#                 u = rename[edge[0]]
#                 v = rename[edge[1]]
#                 if not reverse:
#                     if [u,v] not in edges:
#                         row.append(u)
#                         col.append(v)
#                         edges.append([u,v])
                
#                 if (not directed) or (reverse):
#                     if [v,u] not in edges:
#                         row.append(v)
#                         col.append(u)
#                         rev.append([v,u])
#                         edges.append([v,u])
#                         if v not in revmap:
#                             revmap[v] = []
#                         revmap[v].append(u)
        
        with open(edge_path) as f:
            G1 = nx.DiGraph([[rename[line.split()[0]],rename[line.split()[1]]]
                             for line in f
                             if line.split()[0] in rename and line.split()[1] in rename])
        with open(edge_path) as f:
            G2 = nx.DiGraph([[rename[line.split()[1]],rename[line.split()[0]]]
                             for line in f
                             if line.split()[0] in rename and line.split()[1] in rename])
        G1.remove_edges_from(G1.selfloop_edges())
        G2.remove_edges_from(G2.selfloop_edges())
        row =[]
        col = []
        if not reverse:
            row = row + [e[0] for e in G1.edges()] 
            col = col + [e[1] for e in G1.edges()] 
        if reverse or not directed:
            row = row + [e[0] for e in G2.edges()]
            col = col + [e[1] for e in G2.edges()]
        print(len(row))
#         with open(edge_path) as f:
#             row = [rename[line.split()[0]] for line in f if line.split()[0] in rename and line.split()[1] in rename]
#         with open(edge_path) as f:
#             col = [rename[line.split()[1]] for line in f if line.split()[0] in rename and line.split()[1] in rename]
        edges = [[u,v] for u,v in zip(row,col)]
        
        print('Read edges: DONE')
        edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
#         edge_index, _ = remove_self_loops(edge_index)
#         edge_index, _ = coalesce(edge_index, None, n, n)
        is_rev = []
        print(reverse)
        if not reverse:
            is_rev = is_rev + [0] * len(G1.edges())
        if reverse or not directed:
            is_rev = is_rev + [1] * len(G1.edges())
        print(len(is_rev))
        print(edge_index.shape[1])
        assert (len(is_rev) == edge_index.shape[1])
        
#         print('Annotate edges: RUNNING')
#         is_rev=[]
#         for i in range(edge_index.shape[1]):
#             is_rev.append(1 if edge_index[0,i].item() in revmap and
#                           edge_index[1,i].item() in revmap[edge_index[0,i].item()] else 0)
#         print('Annotate edges: DONE')
        
        data = Data(x=x, edge_index=edge_index, y=y)
        # data.test_mask = torch.tensor(test_ind,dtype=torch.uint8)
        # data.train_mask = torch.tensor(train_ind,dtype=torch.uint8)
        # data.val_mask = torch.tensor(val_ind,dtype=torch.uint8)
        
        data.is_reversed = torch.tensor(is_rev,dtype=torch.uint8) 
        
        return data
