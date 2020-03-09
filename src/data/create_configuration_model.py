import networkx as nx
from networkx.utils import powerlaw_sequence

def generate_conf_model(G,seed=0):
    din=[x[1] for x in G.in_degree()]
    dout=[x[1] for x in G.out_degree()]
    GNoisy=nx.directed_configuration_model(din,dout,create_using=nx.DiGraph(),seed=seed)
    keys = [x[0] for x in G.in_degree()]
    G_mapping = dict(zip(range(len(G.nodes())),keys))
    G_rev_mapping = dict(zip(keys,range(len(G.nodes()))))
    GNoisy = nx.relabel_nodes(GNoisy,G_mapping)
    return GNoisy

def generate_multiple_conf_models(graph_path, output_prefix,output_suffix = '.cites',inits=10):
    fin = graph_path
    G=nx.read_edgelist(fin,create_using=nx.DiGraph())
    for i in range(inits):
        GNoisy = generate_conf_model(G,seed=i)
        fout = f'{output_prefix}_{i}{output_suffix}'
        nx.write_edgelist(GNoisy,fout)


if __name__ == '__main__':
    # generate_multiple_conf_models('data/graphs/processed/cora/cora.cites','data/graphs/confmodel/cora/cora_confmodel')
    # generate_multiple_conf_models('data/graphs/processed/citeseer/citeseer.cites','data/graphs/confmodel/citeseer/citeseer_confmodel')
    # generate_multiple_conf_models('data/graphs/processed/pubmed/pubmed.cites','data/graphs/confmodel/pubmed/pubmed_confmodel')
    # generate_multiple_conf_models('data/graphs/processed/cora_full/cora_full.cites','data/graphs/confmodel/cora_full/cora_full_confmodel')
    generate_multiple_conf_models('data/graphs/processed/cornell/cornell.cites','data/graphs/confmodel/cornell/cornell_confmodel')
    generate_multiple_conf_models('data/graphs/processed/texas/texas.cites','data/graphs/confmodel/texas/texas_confmodel')
    generate_multiple_conf_models('data/graphs/processed/washington/washington.cites','data/graphs/confmodel/washington/washington_confmodel')
    generate_multiple_conf_models('data/graphs/processed/wisconsin/wisconsin.cites','data/graphs/confmodel/wisconsin/wisconsin_confmodel')
    generate_multiple_conf_models('data/graphs/processed/twitter/twitter.cites','data/graphs/confmodel/twitter/twitter_confmodel')
