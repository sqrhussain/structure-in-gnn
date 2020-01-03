import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
import os.path as osp

def read_torch_edgelist(edge_path):
    row, col = [], []
    edges = []
    rev = []
    err = 0
    with open(edge_path, 'r') as f:
        for line in f:
            edge = line.split()
            if edge[0] not in rename or edge[1] not in rename:
                err += 1
                continue
            u = rename[edge[0]]
            v = rename[edge[1]]
            if not self.reverse:
                if [u,v] not in edges:
                    row.append(u)
                    col.append(v)
                    edges.append([u,v])

            if (not self.directed) or (self.reverse):
                if [v,u] not in edges:
                    row.append(v)
                    col.append(u)
                    rev.append([v,u])
                    edges.append([v,u])

    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, n, n)

    is_rev = []
    for i in range(edge_index.shape[1]):
        is_rev.append(1 if edge_index[:,i].numpy().tolist() in rev else 0)
    edge_index = read_torh_edgelist(edge_path)
    data = Data(x=x, edge_index=edge_index)

    data.is_reversed = torch.tensor(is_rev,dtype=torch.uint8)    
    return edge_index
