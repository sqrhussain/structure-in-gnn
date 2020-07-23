
import os
import networkx as nx
import numpy as np

def remove_hubs(dataset,hub_percentage = 1):
    
    print('dataset ' + dataset)
    fin = f'data/graphs/processed/{dataset}/{dataset}.cites'
    G=nx.read_edgelist(fin,create_using=nx.DiGraph())
    n = len(G.nodes())
    m = len(G.edges())
    
    degrees = [G.degree(u) for u in G.nodes]
    deg_threshold = sorted(degrees)[int(n * (1-hub_percentage/100))]

    hubs = [u for u in G.nodes if G.degree(u)>deg_threshold]
    print(f'hubs={len(hubs)}/|V|={n} = {len(hubs)/n:.3f}')
    G.remove_nodes_from(hubs)

    fout = f'data/graphs/removed_hubs/{dataset}/{dataset}_{hub_percentage:02}.cites'
    if not os.path.exists(f'data/graphs/removed_hubs/{dataset}'):
        os.mkdir(f'data/graphs/removed_hubs/{dataset}')
    nx.write_edgelist(G,fout)


if __name__ == '__main__':
    datasets = 'cora citeseer pubmed cora_full twitter webkb'.split()
    for dataset in datasets:
        for percentage in [1,2,4,8]:
            remove_hubs(dataset,percentage)
