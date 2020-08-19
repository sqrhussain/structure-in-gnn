

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
    parser.add_argument('--gen_sbm',
                        type=bool,
                        default=False,
                        help='Should the graph be generated from original labels? Default is False, which means we inject edges to the original graph.')

    # the following are arguments for generating with a constant number of edges, varying degree categories. Only works if --degree_cat is True
    parser.add_argument('--degree_cat',
                        type=bool,
                        default=False,
                        help='Is it a degree category experiment, i.e., adding a constant number of edges but attaching to nodes of varying degree categories? Default is False')
    parser.add_argument('--degree_percentile',
                        type=int,
                        default=5,
                        help='Percentile on which to divide degree in the degree_cat experiment, e.g., 10 means steps of 10%')
    parser.add_argument('--num_injected',
                        type=int,
                        default=500,
                        help='Number of edges to inject in the degree_cat experiment.')

    

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

def get_nodes_with_degree_percentile(G, frm, to):
    # start = min([x[1] for x in G.degree])
    # length = max([x[1] for x in G.degree]) - start

    nodes = sorted(G.degree, key=lambda x: x[1])
    start = 0
    length = len(nodes)
    frm = int(start + length*frm)
    to =  int(start + length*to +1)
    nodes = nodes[frm:to]
    # nodes = [x[0] for x in G.degree if frm<=x[1] and x[1]<to]
    return [x[0] for x in nodes]

def build_label_based_sbm(graph, node_mappings, reverse_node_mappings, labels):
    block_sizes, edge_probabilities, node_lists = build_stochastic_block_matrix(graph, node_mappings, reverse_node_mappings, labels)
    print(edge_probabilities)
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

def get_graph(graph_path, labels, gen_sbm, seed):
    if gen_sbm:
        graph, node_mappings, reverse_node_mappings = create_graph_and_node_mappings_from_file(graph_path)
        edge_probabilities, block_sizes, node_lists = build_label_based_sbm(graph, node_mappings, reverse_node_mappings, labels)

        sbm = stochastic_block_model(block_sizes, edge_probabilities, node_lists, seed, True, False)
        sbm = nx.relabel_nodes(sbm, reverse_node_mappings)
        graph = sbm
    else:
        graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph)
    return graph


def create_label_based_sbm_local_hubs(graph_path, labels, output_path, edges_to_add, seed=0,gen_sbm=False):
    graph = get_graph(graph_path, labels, gen_sbm, seed)

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

def create_label_based_sbm_local_edges(graph_path, labels, output_path, edges_to_add, seed=0,gen_sbm=False):
    graph = get_graph(graph_path, labels, gen_sbm, seed)
    
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


def create_label_based_sbm_distant_edges(graph_path, labels, output_path, edges_to_add, seed=0,gen_sbm=False):
    graph, node_mappings, reverse_node_mappings = create_graph_and_node_mappings_from_file(graph_path)
    
    label_id_to_node_id = create_community_id_to_node_id(labels)

    block_sizes, edge_probabilities, node_lists = build_stochastic_block_matrix(graph, node_mappings, reverse_node_mappings, labels)
    # print(edge_probabilities[0])
    target = [np.argwhere(prob == np.amin(prob)).flatten().tolist() for prob in edge_probabilities]
    
    labels_list = list(label_id_to_node_id.keys())

    graph = get_graph(graph_path, labels, gen_sbm, seed)
    subg = {c:graph.subgraph([u for u in label_id_to_node_id[c]]) for c in labels_list}
    subg_sizes = [len(subg[c].nodes()) for c in labels_list]
    p = [size/len(graph.nodes()) for size in subg_sizes]
    chosen_communities = np.random.choice(labels_list,edges_to_add,replace=True,p=p)

    rows = [np.random.choice(subg[c],1)[0] for c in chosen_communities]
    col_targets = [np.random.choice(target[labels[u]],1)[0] for u in rows]
    # print([labels[u] for u in rows[:10]])
    # print(col_targets[:10])
    # exit(0)
    cols = [np.random.choice(subg[c],1)[0] for c in col_targets]
    edges = [[u,v] for u,v in zip(rows, cols)]

    print(f'{sum([graph.has_edge(*edge) for edge in edges])} edges duplicated')

    graph.add_edges_from(edges)
    nx.write_edgelist(graph, output_path)

    return graph


def create_label_based_sbm_global_hubs(graph_path, labels, output_path, edges_to_add, seed=0,gen_sbm=False):
    graph = get_graph(graph_path, labels, gen_sbm, seed)
        
    label_id_to_node_id = create_community_id_to_node_id(labels)
    labels_list = label_id_to_node_id.keys()

    hubs = get_hubs(graph)
    rows = np.random.choice(hubs,edges_to_add)
    cols = np.random.choice(graph.nodes(),edges_to_add)
    edges = [[u,v] for u,v in zip(rows, cols)]

    graph.add_edges_from(edges)
    nx.write_edgelist(graph, output_path)

    return graph



def create_label_based_sbm_global_degree_cat(graph_path, labels, output_path, edges_to_add, frm, to, seed=0,gen_sbm=False):
    graph = get_graph(graph_path, labels, gen_sbm, seed)
        
    label_id_to_node_id = create_community_id_to_node_id(labels)
    labels_list = label_id_to_node_id.keys()

    attach_to = get_nodes_with_degree_percentile(graph,frm,to)
    print(f'Percentage ({frm}-{to}) able to link to {len(attach_to)} nodes')
    rows = np.random.choice(attach_to,edges_to_add)
    cols = np.random.choice(graph.nodes(),edges_to_add)
    edges = [[u,v] for u,v in zip(rows, cols)]

    graph.add_edges_from(edges)
    nx.write_edgelist(graph, output_path)

    return graph

def create_label_based_sbm_global_edges(graph_path, labels, output_path, edges_to_add, seed=0,gen_sbm=False):
    graph = get_graph(graph_path, labels, gen_sbm, seed)

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
    if args.degree_cat:
        for dataset in args.datasets:
            graph_path = f'data/graphs/processed/{dataset}/{dataset}.cites'
            output_dir = f'data/graphs/injected_edges_degree_cat/{dataset}'
            labels_path = f'data/community_id_dicts/{dataset}/{dataset}_louvain.pickle'
            labels = load_communities(labels_path)
            print(f'graph: {graph_path}')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            for percentile in range(0,100,args.degree_percentile):
                for seed in range(args.runs):
                    output_path = output_dir + f'/{dataset}_global_edges_{args.num_injected}_{seed}_{percentile}_to_{percentile+args.degree_percentile}.cites'
                    print(f'output: {output_path}')
                    frm = percentile/100
                    to =(percentile+args.degree_percentile)/100
                    create_label_based_sbm_global_degree_cat(graph_path, labels, output_path, args.num_injected, frm, to, seed=seed, gen_sbm=False)
    else:
        edges = range(6000,10001,1000)
        for dataset in args.datasets:
            for edge in edges:
                for seed in range(args.runs):
                    graph_path = f'data/graphs/processed/{dataset}/{dataset}.cites'
                    if args.gen_sbm:
                        labels_path = f'data/graphs/processed/{dataset}/{dataset}.content'
                        labels,_ = load_labels(labels_path)
                        output_dir = f'data/graphs/injected_edges_sbm/{dataset}'
                    else:
                        labels_path = f'data/community_id_dicts/{dataset}/{dataset}_louvain.pickle'
                        labels = load_communities(labels_path)
                        output_dir = f'data/graphs/injected_edges/{dataset}'
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    output_path = output_dir + f'/{dataset}_local_hubs_{edge}_{seed}.cites'
                    if not os.path.exists(output_path):
                        create_label_based_sbm_local_hubs(graph_path, labels, output_path, edges_to_add=edge, seed=seed, gen_sbm=args.gen_sbm)

                    output_path = output_dir + f'/{dataset}_local_edges_{edge}_{seed}.cites'
                    if not os.path.exists(output_path):
                        create_label_based_sbm_local_edges(graph_path, labels, output_path, edges_to_add=edge, seed=seed, gen_sbm=args.gen_sbm)

                    output_path = output_dir + f'/{dataset}_global_hubs_{edge}_{seed}.cites'
                    if not os.path.exists(output_path):
                        create_label_based_sbm_global_hubs(graph_path, labels, output_path, edges_to_add=edge, seed=seed, gen_sbm=args.gen_sbm)

                    output_path = output_dir + f'/{dataset}_global_edges_{edge}_{seed}.cites'
                    if not os.path.exists(output_path):
                        create_label_based_sbm_global_edges(graph_path, labels, output_path, edges_to_add=edge, seed=seed, gen_sbm=args.gen_sbm)

                    # output_path = output_dir + f'/{dataset}_distant_edges_{edge}_{seed}.cites'
                    # if not os.path.exists(output_path):
                    #     create_label_based_sbm_distant_edges(graph_path, labels, output_path, edges_to_add=edge, seed=seed, gen_sbm=args.gen_sbm)
if __name__ == '__main__':
    main()