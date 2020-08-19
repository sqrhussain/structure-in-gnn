import networkx as nx
import numpy as np
import community as comms
import pickle
import os
from networkx.generators.community import stochastic_block_model
from scipy.sparse import csr_matrix


def create_graph_and_node_mappings_from_file(filepath):
    G = nx.DiGraph()
    node_mappings = {}
    reverse_node_mappings = {}
    with open(filepath, 'r') as f:
        counter = 0
        for line in f:
            edge = line.strip().split()
            node1 = str(edge[0])
            node2 = str(edge[1])
            if node1 not in node_mappings:
                node_mappings[node1] = counter
                reverse_node_mappings[counter] = node1
                counter += 1
            if node2 not in node_mappings:
                node_mappings[node2] = counter
                reverse_node_mappings[counter] = node2
                counter += 1
            source_node = node_mappings[node1]
            target_node = node_mappings[node2]
            G.add_edge(source_node, target_node)
    return G, node_mappings, reverse_node_mappings


def create_louvain_communities_dict(G, reverse_node_mappings):
    G = nx.to_undirected(G)
    node_communities_louvain = comms.best_partition(G)
    node_communities_louvain_original_ids = {}
    distinct_communities = []
    for reverse_node_id in node_communities_louvain:
        node_communities_louvain_original_ids[reverse_node_mappings[reverse_node_id]] = node_communities_louvain[
            reverse_node_id]
        if node_communities_louvain[reverse_node_id] not in distinct_communities:
            distinct_communities.append(node_communities_louvain[reverse_node_id])
    return node_communities_louvain_original_ids


def store_in_file(community_dict, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(community_dict, handle, protocol=2)


def load_from_file(path):
    with open(path, 'rb') as handle:
        ret = pickle.load(handle)
    return ret


def calculate_edge_probabilities(G, communities, node_id_community_id_dict, reverse_node_mappings=None):
    between_community_stats = {}
    for edge in G.edges():
        # set source and target id
        source_id = edge[0]
        target_id = edge[1]
        if reverse_node_mappings is not None:
            source_id = reverse_node_mappings[source_id]
            target_id = reverse_node_mappings[target_id]
        source_community_id = node_id_community_id_dict[source_id]
        target_community_id = node_id_community_id_dict[target_id]
        if (source_community_id, target_community_id) not in between_community_stats:
            source_community_size = len(communities[source_community_id])
            target_community_size = len(communities[target_community_id])
            if G.is_directed():
                if source_community_id == target_community_id:
                    max_edge_count = (source_community_size * (source_community_size - 1))
                    if has_selfloops(G):
                        max_edge_count += source_community_size
                else:
                    max_edge_count = 2 * source_community_size * target_community_size
            else:
                if source_community_id == target_community_id:
                    max_edge_count = (source_community_size * (source_community_size - 1)) / 2
                    if has_selfloops(G):
                        max_edge_count += source_community_size
                else:
                    max_edge_count = source_community_size * target_community_size
            between_community_stats[(source_community_id, target_community_id)] = {"existing_edge_count": 0,
                                                                                   "max_edge_count": max_edge_count}
            if not G.is_directed() and source_community_id != target_community_id:
                between_community_stats[(target_community_id, source_community_id)] = {"existing_edge_count": 0,
                                                                                       "max_edge_count": max_edge_count}
        between_community_stats[(source_community_id, target_community_id)]['existing_edge_count'] += 1
        if not G.is_directed() and source_community_id != target_community_id:
            between_community_stats[(target_community_id, source_community_id)]['existing_edge_count'] += 1
    for key in between_community_stats:
        between_community_stats[key]['edge_probability'] = between_community_stats[key]['existing_edge_count'] / \
                                                           between_community_stats[key]['max_edge_count']
    rows = []
    cols = []
    data = []
    for key in between_community_stats:
        rows.append(key[0])
        cols.append(key[1])
        if between_community_stats[key]['edge_probability'] < 0 or between_community_stats[key]['edge_probability'] > 1:
            print(key)
        data.append(between_community_stats[key]['edge_probability'])
    communities_count = len(communities)
    return csr_matrix((data, (rows, cols)), shape=(communities_count, communities_count),
                      dtype=float).todense().tolist()


def has_selfloops(G):
    return nx.number_of_selfloops(G) > 0


def get_block_sizes(communities):
    block_sizes = []
    for community_id in communities:
        block_sizes.append(len(communities[community_id]))
    return block_sizes


def get_nodelist(communities, node_mappings=None):
    nodelist = []
    for community_id in communities:
        community = communities[community_id]
        for node_id in community:
            if node_mappings is not None:
                nodelist.append(node_mappings[node_id])
            else:
                nodelist.append(node_id)
    return nodelist


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def create_community_id_to_node_id(node_communities):
    community_id_to_node_id = {}
    for node_id in node_communities:
        community_id = node_communities[node_id]
        if community_id not in community_id_to_node_id:
            community_id_to_node_id[community_id] = []
        community_id_to_node_id[community_id].append(node_id)
    return community_id_to_node_id


def build_stochastic_block_matrix(graph, node_mappings, reverse_node_mappings, node_communities):
    community_id_to_node_id = create_community_id_to_node_id(node_communities)
    edge_probabilities = calculate_edge_probabilities(graph, community_id_to_node_id,
                                                      node_communities, reverse_node_mappings)
    block_sizes = get_block_sizes(community_id_to_node_id)
    nodelists = get_nodelist(community_id_to_node_id,  node_mappings)

    return block_sizes, edge_probabilities, nodelists



def create_sbm_graph(graph, block_sizes, edge_probabilities, node_lists, output_path, seed, reverse_node_mappings):
    sbm = stochastic_block_model(block_sizes, edge_probabilities, node_lists, seed, True, has_selfloops(graph))
    sbm = nx.relabel_nodes(sbm, reverse_node_mappings)
    nx.write_edgelist(sbm, output_path)


def load_communities(community_path,graph=None,reverse_node_mappings=None):
    # community detection
    if not os.path.exists(community_path):
        if graph is None or reverse_node_mappings is None:
            print('graph or reverse_node_mappings is None')
            raise
        print('Detecting communities')
        node_communities_louvain = create_louvain_communities_dict(graph, reverse_node_mappings)
        store_in_file(node_communities_louvain, community_path)
    else:
        node_communities_louvain = load_from_file(community_path)
    return node_communities_louvain

def create_multiple_sbm_graphs(graph_path, community_path, output_prefix, output_suffix='.cites', inits=10):

    graph, node_mappings, reverse_node_mappings = create_graph_and_node_mappings_from_file(graph_path)
    # community detection
    node_communities_louvain = load_communities(community_path,graph,reverse_node_mappings)
    
    # build stochastic matrix
    block_sizes, edge_probabilities, node_lists = build_stochastic_block_matrix(graph, node_mappings, reverse_node_mappings, node_communities_louvain)
    for i in range(inits):
        create_sbm_graph(graph, block_sizes, edge_probabilities, node_lists,
                         f'{output_prefix}_{i}{output_suffix}', i, reverse_node_mappings)


if __name__ == "__main__":
    # create_multiple_sbm_graphs('data/graphs/processed/cora/cora.cites',
    #                            'data/community_id_dicts/cora/cora_louvain.pickle',
    #                            'data/graphs/sbm/cora/cora_sbm')
    # create_multiple_sbm_graphs('data/graphs/processed/citeseer/citeseer.cites',
    #                            'data/community_id_dicts/citeseer/citeseer_louvain.pickle',
    #                            'data/graphs/sbm/citeseer/citeseer_sbm')
    # create_multiple_sbm_graphs('data/graphs/processed/pubmed/pubmed.cites',
    #                            'data/community_id_dicts/pubmed/pubmed_louvain.pickle',
    #                            'data/graphs/sbm/pubmed/pubmed_sbm')
    # create_multiple_sbm_graphs('data/graphs/processed/cora_full/cora_full.cites',
    #                            'data/community_id_dicts/cora_full/cora_full_louvain.pickle',
    #                            'data/graphs/sbm/cora_full/cora_full_sbm')

    # create_multiple_sbm_graphs('data/graphs/processed/cornell/cornell.cites',
    #                            'data/community_id_dicts/cornell/cornell_louvain.pickle',
    #                            'data/graphs/sbm/cornell/cornell_sbm')
    # create_multiple_sbm_graphs('data/graphs/processed/texas/texas.cites',
    #                            'data/community_id_dicts/texas/texas_louvain.pickle',
    #                            'data/graphs/sbm/texas/texas_sbm')
    # create_multiple_sbm_graphs('data/graphs/processed/washington/washington.cites',
    #                            'data/community_id_dicts/washington/washington_louvain.pickle',
    #                            'data/graphs/sbm/washington/washington_sbm')
    # create_multiple_sbm_graphs('data/graphs/processed/wisconsin/wisconsin.cites',
    #                            'data/community_id_dicts/wisconsin/wisconsin_louvain.pickle',
    #                            'data/graphs/sbm/wisconsin/wisconsin_sbm')

    # create_multiple_sbm_graphs('data/graphs/processed/twitter/twitter.cites',
    #                            'data/community_id_dicts/twitter/twitter_louvain.pickle',
    #                            'data/graphs/sbm/twitter/twitter_sbm')

    # create_multiple_sbm_graphs('data/graphs/processed/webkb/webkb.cites',
    #                            'data/community_id_dicts/webkb/webkb_louvain.pickle',
    #                            'data/graphs/sbm/webkb/webkb_sbm')

    # create_multiple_sbm_graphs('data/graphs/processed/amazon_electronics_computers/amazon_electronics_computers.cites',
    #                            'data/community_id_dicts/amazon_electronics_computers/amazon_electronics_computers_louvain.pickle',
    #                            'data/graphs/sbm/amazon_electronics_computers/amazon_electronics_computers_sbm')

    # create_multiple_sbm_graphs('data/graphs/processed/amazon_electronics_photo/amazon_electronics_photo.cites',
    #                            'data/community_id_dicts/amazon_electronics_photo/amazon_electronics_photo_louvain.pickle',
    #                            'data/graphs/sbm/amazon_electronics_photo/amazon_electronics_photo_sbm')

    # create_multiple_sbm_graphs('data/graphs/processed/ms_academic_cs/ms_academic_cs.cites',
    #                            'data/community_id_dicts/ms_academic_cs/ms_academic_cs_louvain.pickle',
    #                            'data/graphs/sbm/ms_academic_cs/ms_academic_cs_sbm')

    # create_multiple_sbm_graphs('data/graphs/processed/ms_academic_phy/ms_academic_phy.cites',
    #                            'data/community_id_dicts/ms_academic_phy/ms_academic_phy_louvain.pickle',
    #                            'data/graphs/sbm/ms_academic_phy/ms_academic_phy_sbm')
    create_multiple_sbm_graphs('data/graphs/processed/wiki_cs/wiki_cs.cites',
                               'data/community_id_dicts/wiki_cs/wiki_cs_louvain.pickle',
                               'data/graphs/sbm/wiki_cs/wiki_cs_sbm')


