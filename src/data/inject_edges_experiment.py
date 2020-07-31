

from src.data.create_stochastic_block_model import create_graph_and_node_mappings_from_file, build_stochastic_block_matrix, load_communities
from src.data.create_stochastic_block_model import create_community_id_to_node_id, create_sbm_graph
import pandas as pd
import numpy as np
import networkx as nx
from networkx.generators.community import stochastic_block_model
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Create datasets with injected edges between/within communities using hubs/normal nodes.")

    parser.add_argument('--datasets',
                        nargs='+',
                        help='datasets to process, e.g., --dataset cora pubmed')
    parser.add_argument('--runs',
                        type=int,
                        default=5,
                        help='Number of random initializations. Default is 5.')
    return parser.parse_args()

def load_labels(path):
    label = {}
    cnt = 0
    label_mapping = {}
    reverse_label_mapping = {}
    with open(path, 'r') as handle:
        label = {}
        for line in handle:
            s = line.strip().split()
            if s[-1] not in label_mapping:
                label_mapping[s[-1]] = cnt
                reverse_label_mapping[cnt] = s[-1]
                cnt+=1
            label[s[0]] = label_mapping[s[-1]]
    return label,reverse_label_mapping

	

# def label_stochastic_matrix(graph_path, labels_path):
#     graph, node_mappings, reverse_node_mappings = create_graph_and_node_mappings_from_file(graph_path)
#     labels,remap = load_labels(labels_path)
#     node_labels = {}
#     for node in labels:
#         node_labels[node_mappings[node]] = labels[node]
#     label_id_to_node_id = create_community_id_to_node_id(node_labels)
#     edge_probabilities = calculate_edge_probabilities(graph, label_id_to_node_id,
#                                                       node_labels, None)
#     return edge_probabilities,remap


def get_hubs(G):
	# TODO: define a way to find the constant instead of always using 0.01
	nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
	nodes = nodes[:int(np.ceil(0.01*len(nodes)))]
	return [x[0] for x in nodes]

def build_label_based_sbm(graph, node_mappings, reverse_node_mappings, labels):
    block_sizes, edge_probabilities, node_lists = build_stochastic_block_matrix(graph, node_mappings, reverse_node_mappings, labels)
    n = len(block_sizes)
    avg_node_degree = 2*len(graph.edges)/len(graph.nodes)
    diag = np.diag((avg_node_degree*np.array(block_sizes)/2)/np.array(block_sizes)/np.array(block_sizes))
    edge_probabilities = diag/2 + np.ones((n,n))*0.001
    return edge_probabilities, block_sizes, node_lists

def create_label_based_sbm(graph_path, labels, output_path, seed=0):
    graph, node_mappings, reverse_node_mappings = create_graph_and_node_mappings_from_file(graph_path)
    node_labels = {}
    for node in labels:
        node_labels[node_mappings[node]] = labels[node]
    edge_probabilities, block_sizes, node_lists = build_label_based_sbm(graph, node_mappings, reverse_node_mappings, labels)
    create_sbm_graph(graph, block_sizes, edge_probabilities, node_lists, output_path, seed, reverse_node_mappings)
    return edge_probabilities



def create_label_based_sbm_local_hubs(graph_path, labels, output_path, edges_to_add, seed=0):
    graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph)
    
    label_id_to_node_id = create_community_id_to_node_id(labels)
    labels_list = list(label_id_to_node_id.keys())

    subg = {c:graph.subgraph([u for u in label_id_to_node_id[c]]) for c in labels_list}
    subg_sizes = [len(subg[c].nodes()) for c in labels_list]
    p = [size/len(graph.nodes()) for size in subg_sizes]

    chosen_communities = np.random.choice(labels_list,edges_to_add,replace=True,p=p)
    community_hubs = {c:get_hubs(subg[c]) for c in chosen_communities}

    rows = [np.random.choice(community_hubs[c],1)[0] for c in chosen_communities]
    cols = [np.random.choice(subg[c],1)[0] for c in chosen_communities]
    edges = [[u,v] for u,v in zip(rows, cols)]

    print(f'{sum([graph.has_edge(*edge) for edge in edges])} edges duplicated')

    graph.add_edges_from(edges)
    nx.write_edgelist(graph, output_path)

    return graph
    # # constants to be defined later
    # hubs_ratio = 0.01
    # hubs_connectivity = 0.30
    
    # # add local hubs
    # edges = []
    # for label in label_id_to_node_id:
    #     nodes = label_id_to_node_id[label]
    #     # pick a couple of nodes
    #     n_hubs = int(len(nodes) * hubs_ratio) + 1
    #     m_hubs = int(len(nodes) * hubs_connectivity) + 1
    #     hubs = np.random.choice(nodes,n_hubs)
        
    #     for hub in hubs:
    #         neighbors = np.random.choice(nodes,m_hubs)
    #         edges = edges + [[hub,u] for u in neighbors]
    # sbm.add_edges_from(edges)
    # nx.write_edgelist(sbm, output_path)
    # return edge_probabilities

def create_label_based_sbm_local_edges(graph_path, labels, output_path, edges_to_add, seed=0):
    graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph)
    
    label_id_to_node_id = create_community_id_to_node_id(labels)
    labels_list = list(label_id_to_node_id.keys())

    subg = {c:graph.subgraph([u for u in label_id_to_node_id[c]]) for c in labels_list}
    subg_sizes = [len(subg[c].nodes()) for c in labels_list]
    p = [size/len(graph.nodes()) for size in subg_sizes]

    chosen_communities = np.random.choice(labels_list,edges_to_add,replace=True,p=p)

    rows = [np.random.choice(subg[c],1)[0] for c in chosen_communities]
    cols = [np.random.choice(subg[c],1)[0] for c in chosen_communities]
    edges = [[u,v] for u,v in zip(rows, cols)]

    print(f'{sum([graph.has_edge(*edge) for edge in edges])} edges duplicated')

    graph.add_edges_from(edges)
    nx.write_edgelist(graph, output_path)

    return graph




def create_label_based_sbm_global_hubs(graph_path, labels, output_path, edges_to_add, seed=0):
    graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph)
        
    label_id_to_node_id = create_community_id_to_node_id(labels)
    labels_list = label_id_to_node_id.keys()

    hubs = get_hubs(graph)
    rows = np.random.choice(hubs,edges_to_add)
    cols = np.random.choice(graph.nodes(),edges_to_add)
    edges = [[u,v] for u,v in zip(rows, cols)]

    graph.add_edges_from(edges)
    nx.write_edgelist(graph, output_path)

    return graph


def create_label_based_sbm_global_edges(graph_path, labels, output_path, edges_to_add, seed=0):
    graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph)

    label_id_to_node_id = create_community_id_to_node_id(labels)
    labels_list = label_id_to_node_id.keys()

    rows = np.random.choice(graph.nodes(),edges_to_add)
    cols = np.random.choice(graph.nodes(),edges_to_add)
    edges = [[u,v] for u,v in zip(rows, cols)]

    graph.add_edges_from(edges)
    nx.write_edgelist(graph, output_path)

    return graph



def main():
    args = parse_args()


    # labels,remap = load_labels(labels_path)
    # labels = load_communities(labels_path)
    edges = range(100,2001,100)
    for dataset in args.datasets:
    	for edge in edges:
    		for seed in range(args.runs):
    			graph_path = f'data/graphs/processed/{dataset}/{dataset}.cites'
    			output_dir = f'data/graphs/injected_edges/{dataset}'
    			labels_path = f'data/community_id_dicts/{dataset}/{dataset}_louvain.pickle'
    			if not os.path.exists(output_dir):
    				os.mkdir(output_dir)
    			output_path = output_dir + f'/{dataset}_local_hubs_{edge}_{seed}.cites'
    			create_label_based_sbm_local_hubs(graph_path, load_communities(labels_path), output_path, edges_to_add=edge, seed=seed)

    			output_path = output_dir + f'/{dataset}_local_edges_{edge}_{seed}.cites'
    			create_label_based_sbm_local_edges(graph_path, load_communities(labels_path), output_path, edges_to_add=edge, seed=seed)

    			output_path = output_dir + f'/{dataset}_global_hubs_{edge}_{seed}.cites'
    			create_label_based_sbm_global_hubs(graph_path, load_communities(labels_path), output_path, edges_to_add=edge, seed=seed)

    			output_path = output_dir + f'/{dataset}_global_edges_{edge}_{seed}.cites'
    			create_label_based_sbm_global_edges(graph_path, load_communities(labels_path), output_path, edges_to_add=edge, seed=seed)

if __name__ == '__main__':
	main()