import networkx as nx
import numpy as np
import os

def add_2hop_edges(dataset,percentage=1,seed=0):   
    # assert percentage <= 100 and percentage >= 0
    assert type(percentage) is int
    
    fin = f'data/graphs/processed/{dataset}/{dataset}.cites'
    g = nx.read_edgelist(fin,create_using=nx.DiGraph())
    ug = g.to_undirected()
    
    nodes = np.array(list(g.nodes()))
    
    np.random.seed(seed)
    node_idx = np.random.choice(len(nodes),int(len(nodes)*percentage/100),replace=True)
    
    def create_2hop_edge(graph, src):
        u = src
        while u==src or u in graph.neighbors(src):
            potential = list(graph.neighbors(u))
            u = np.random.choice(potential)
        return [src,u]
    
    src_nodes = nodes[node_idx]

#     edges_to_add = [create_2hop_edge(ug,u) for u in src_nodes] # could create duplicates
    edges_to_add = []
    for u in src_nodes:
        new_edge = create_2hop_edge(ug,u)
        edges_to_add.append(new_edge)
        ug.add_edge(new_edge[0],new_edge[1])
        
    g.add_edges_from(edges_to_add)
    outdir_path = f'data/graphs/added_2hop_edges/{dataset}'
    if not os.path.exists(outdir_path):
        os.mkdir(outdir_path)
    outfile_name = f'{dataset}_{percentage:02}.cites'
    outfile_path = f'{outdir_path}/{outfile_name}'
    
    nx.write_edgelist(g,outfile_path)
    


if __name__ == '__main__':

    datasets = datasets = 'cora citeseer twitter'.split()
    for dataset in datasets:
        for percentage in [64,128,256,512]:
            add_2hop_edges(dataset,percentage)