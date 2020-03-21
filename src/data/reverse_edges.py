
import os
import networkx as nx
import numpy as np

def flip_edges(dataset,percentage=10,seed=0):
    assert percentage <= 100 and percentage >= 0
    assert type(percentage) is int
    
    fin = f'data/graphs/processed/{dataset}/{dataset}.cites'
    
    g = nx.read_edgelist(fin,create_using=nx.DiGraph())
    
    edges = np.array(list(g.edges()))
    np.random.seed(seed)
    flip_idx = np.random.choice(len(edges),int(len(edges)*percentage/100))
    
    edges_to_flip = edges[flip_idx].tolist()
    flipped_edges = [[u[1],u[0]] for u in edges_to_flip]

    g.remove_edges_from(edges_to_flip)
    g.add_edges_from(flipped_edges)
    
    outdir_path = f'data/graphs/flipped_edges/{dataset}'
    if not os.path.exists(outdir_path):
        os.mkdir(outdir_path)
    outfile_name = f'{dataset}_{percentage}.cites'
    outfile_path = f'{outdir_path}/{outfile_name}'
    
    nx.write_edgelist(g,outfile_path)
    

if __name__ == '__main__':
    datasets = 'cora citeseer pubmed cora_full twitter webkb'.split()
    for dataset in datasets:
        for percentage in range(10,101,10):
            flip_edges(dataset,percentage)