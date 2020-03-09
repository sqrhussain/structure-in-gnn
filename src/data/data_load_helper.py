import torch
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
import pandas as pd
from random import sample, seed
import numpy as np
from torch_geometric.data import Data
import networkx as nx

def read_network(features_path,edge_path,directed,reverse, convert_to_BoW = False):
    if not directed:
        reverse = False
    feats = []
    target = []
    rename = {}
    class_rename = {}
    cnt = 0
    class_cnt = 0
    print('Read features: RUNNING')
    with open(features_path, 'r') as f:
        for line in f:
            info = line.split()
            if len(info) == 1:
                info = line.split(',')
            rename[info[0]] = cnt
            feats.append(np.array([float(x) for x in info[1:-1]]))
            if info[-1] not in class_rename:
                class_rename[info[-1]] = class_cnt
                class_cnt+=1
            target.append(class_rename[info[-1]])
            cnt += 1
    # TF-IDF to binary BoW
    if convert_to_BoW:
        feats = (np.array(feats) > 0)
    else:
        feats = np.array(feats)
    y = torch.tensor(target,dtype=torch.long)
    n = len(target)

    x = torch.tensor(np.array(feats), dtype=torch.float)
    
    print('Read features: DONE')
    # 3. Split similar to Planetoid
    num_classes = len(set(target))
    df = pd.DataFrame(target)
    df.columns = ['target']
    
    print('Read edges: RUNNING')
    # 4. Read edges
    
    with open(edge_path) as f:
        G1 = nx.DiGraph([[rename[line.split()[0]],rename[line.split()[1]]]
                         for line in f
                         if line.split()[0] in rename and line.split()[1] in rename])
    with open(edge_path) as f:
        G2 = nx.DiGraph([[rename[line.split()[1]],rename[line.split()[0]]]
                         for line in f
                         if line.split()[0] in rename and line.split()[1] in rename])
    G1.remove_edges_from(nx.selfloop_edges(G1))
    G2.remove_edges_from(nx.selfloop_edges(G2))
    row = []
    col = []
    if not reverse:
        row = row + [e[0] for e in G1.edges()] 
        col = col + [e[1] for e in G1.edges()] 
    if reverse or not directed:
        row = row + [e[0] for e in G2.edges()]
        col = col + [e[1] for e in G2.edges()]
    print('Read edges: DONE')
    print(f' {len(row)} edges')
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    is_rev = []
    if not reverse:
        is_rev = is_rev + [0] * len(G1.edges())
    if reverse or not directed:
        is_rev = is_rev + [1] * len(G1.edges())

    assert (len(is_rev) == edge_index.shape[1])
            
    data = Data(x=x, edge_index=edge_index, y=y)
    
    data.is_reversed = torch.tensor(is_rev,dtype=torch.uint8) 
    
    return data




def load_embedding(embFile,featFile = None):
    xDict = {}
    yDict = {}
    
    rename_class = {}
    cnt_class = 0
    with open(featFile, 'r') as f:
        for line in f:
            s = line.split()
            id = s[0]
            emb = [float(x) for x in s[1:-1]]
            xDict[id] = emb
            if s[-1] not in rename_class:
                rename_class[s[-1]] = cnt_class
                cnt_class += 1
            yDict[id] = rename_class[s[-1]]

    rename_id={}
    cnt = 0
    for k in xDict.keys():
        rename_id[cnt] = k
        cnt += 1
            
    # Read embedding
    if embFile is not None:
        err = 0
        with open(embFile, 'r') as f:
            skip = True
            for line in f:
                s = line.split()
                if skip and len(s)==2:
                    skip = False
                    continue
                skip = False
                id = s[0]
                emb = [float(x) for x in s[1:]]
                if id in xDict:
                    xDict[id] = xDict[id] + emb
                else:
                    err +=1
        if err>0:
            print(f'WARNING: {err} items have no embedding generated')

    
    x = []
    target = []
    for i in range(cnt):
        x.append(np.array(xDict[rename_id[i]]))
        target.append(yDict[rename_id[i]])

    
    x = (np.array(x)>0)
    x = torch.tensor(x,dtype=torch.float)
    y = torch.tensor(np.array(target),dtype=torch.long)
    
    data = Data(x=x, y=y)
    
    return data


