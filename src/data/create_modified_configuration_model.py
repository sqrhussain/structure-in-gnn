import networkx as nx
from networkx.utils import powerlaw_sequence
import os
from src.data.create_stochastic_block_model import load_labels

def generate_conf_model(G,seed=0):
    din=[x[1] for x in G.in_degree()]
    dout=[x[1] for x in G.out_degree()]
    GNoisy=nx.directed_configuration_model(din,dout,create_using=nx.DiGraph(),seed=seed)
    keys = [x[0] for x in G.in_degree()]
    G_mapping = dict(zip(range(len(G.nodes())),keys))
    G_rev_mapping = dict(zip(keys,range(len(G.nodes()))))
    GNoisy = nx.relabel_nodes(GNoisy,G_mapping)
    return GNoisy


def generate_modified_conf_model(G, seed=0):
    node_labels_dict = nx.get_node_attributes(G,'label')
    unique_node_labels = set(node_labels_dict.values())
    same_label_subgraphs = {}
    for node_label in unique_node_labels:
        same_label_subgraphs[node_label] = nx.DiGraph()
    edges_to_remove = []
    for edge in G.edges:
        if node_labels_dict[edge[0]] == node_labels_dict[edge[1]]:
            node_label = G.nodes(data=True)[edge[0]]['label']
            same_label_subgraphs[node_label].add_edge(edge[0], edge[1])
            edges_to_remove.append((edge[0], edge[1]))
    G.remove_edges_from(edges_to_remove)
    for label in same_label_subgraphs:
        G.add_edges_from(generate_conf_model(same_label_subgraphs[label], seed).edges)
    return G


def generate_multiple_modified_conf_models(graph_path, content_path, output_prefix,output_suffix = '.cites',inits=10):
    fin = graph_path
    G=nx.read_edgelist(fin,create_using=nx.DiGraph())
    node_labels_dict,_ = load_labels(content_path)
    nx.set_node_attributes(G,node_labels_dict, name='label')
    for i in range(inits):
        print(f'{graph_path} ... {output_prefix}_{i}{output_suffix}')
        GNoisy = generate_modified_conf_model(G,seed=i)
        fout = f'{output_prefix}_{i}{output_suffix}'
        nx.write_edgelist(GNoisy,fout)



if __name__ == '__main__':
    datasets = 'cora citeseer twitter chameleon squirrel webkb actor pubmed cora_full'.split()
    for dataset in datasets:

        if not os.path.exists(f'data/graphs/modcm/{dataset}/'):
            os.mkdir(f'data/graphs/modcm/{dataset}/')
        generate_multiple_modified_conf_models(f'data/graphs/processed/{dataset}/{dataset}.cites',
                                f'data/graphs/processed/{dataset}/{dataset}.content',
                                f'data/graphs/modcm/{dataset}/{dataset}_modcm')
    # generate_multiple_modified_conf_models('data/graphs/processed/cora/cora.cites','data/graphs/confmodel/cora/cora_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/citeseer/citeseer.cites','data/graphs/confmodel/citeseer/citeseer_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/pubmed/pubmed.cites','data/graphs/confmodel/pubmed/pubmed_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/cora_full/cora_full.cites','data/graphs/confmodel/cora_full/cora_full_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/cornell/cornell.cites','data/graphs/confmodel/cornell/cornell_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/texas/texas.cites','data/graphs/confmodel/texas/texas_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/washington/washington.cites','data/graphs/confmodel/washington/washington_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/wisconsin/wisconsin.cites','data/graphs/confmodel/wisconsin/wisconsin_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/twitter/twitter.cites','data/graphs/confmodel/twitter/twitter_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/webkb/webkb.cites','data/graphs/confmodel/webkb/webkb_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/amazon_electronics_computers/amazon_electronics_computers.cites',
    #                     'data/graphs/confmodel/amazon_electronics_computers/amazon_electronics_computers_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/amazon_electronics_photo/amazon_electronics_photo.cites',
    #                     'data/graphs/confmodel/amazon_electronics_photo/amazon_electronics_photo_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/ms_academic_cs/ms_academic_cs.cites',
    #                     'data/graphs/confmodel/ms_academic_cs/ms_academic_cs_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/ms_academic_phy/ms_academic_phy.cites',
    #                     'data/graphs/confmodel/ms_academic_phy/ms_academic_phy_confmodel')
    # generate_multiple_modified_conf_models('data/graphs/processed/wiki_cs/wiki_cs.cites',
    #                     'data/graphs/confmodel/wiki_cs/wiki_cs_confmodel')
