import networkx as nx
import os

def get_random_graph(n,m,seed=0):
	d = m//n
	if (n*d)%2!=0:
		d+=1
	return nx.random_regular_graph(d,n,seed=seed)

def get_erdos_renyi_graph(n,m,seed=0):
	p = m / (n*(n-1)/2)
	return nx.gnp_random_graph(n,p,seed)

def generate_random_graph(dataset,seed=0):
	fin = f'data/graphs/processed/{dataset}/{dataset}.cites'
	fout = f'data/graphs/erdos/{dataset}/{dataset}_{seed}.cites'
	if not os.path.exists(f'data/graphs/erdos/{dataset}'):
		os.mkdir(f'data/graphs/erdos/{dataset}')
	G=nx.read_edgelist(fin,create_using=nx.DiGraph())
	relabel = {i:u for i,u in enumerate(G.nodes())}
	GNoisy = get_erdos_renyi_graph(len(G.nodes()),len(G.edges()),seed)
	GNoisy = nx.relabel_nodes(GNoisy,relabel)
	nx.write_edgelist(GNoisy,fout)



if __name__=='__main__':
	inits = 10
	if not os.path.exists('data/graphs/erdos'):
		os.mkdir('data/graphs/erdos')
	for dataset in 'wiki_cs'.split():
		for seed in range(inits):
			generate_random_graph(dataset,seed=seed)