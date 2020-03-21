import networkx as nx
import os

def get_random_graph(n,m):
	d = m//n
	if (n*d)%2!=0:
		d+=1
	return nx.random_regular_graph(d,n,seed=0)

def generate_random_graph(dataset):
	fin = f'data/graphs/processed/{dataset}/{dataset}.cites'
	fout = f'data/graphs/random/{dataset}/{dataset}.cites'
	if not os.path.exists(f'data/graphs/random/{dataset}'):
		os.mkdir(f'data/graphs/random/{dataset}')
	G=nx.read_edgelist(fin,create_using=nx.DiGraph())
	relabel = {i:u for i,u in enumerate(G.nodes())}
	GNoisy = get_random_graph(len(G.nodes()),len(G.edges()))
	GNoisy = nx.relabel_nodes(GNoisy,relabel)
	nx.write_edgelist(GNoisy,fout)



if __name__=='__main__':
	if not os.path.exists('data/graphs/random'):
		os.mkdir('data/graphs/random')
	for dataset in 'cora citeseer pubmed cora_full twitter webkb'.split():
		generate_random_graph(dataset)