


from subprocess import call
import os

inits = 20
negs = [20]
rhos = [0.025]

def run_nerd(dataset,outfile1,outfile2,neg,rho,size=128):
	if os.path.exists(outfile1) and os.path.exists(outfile2):
		print(f'{outfile1} and {outfile2} already exist!')
		return 
	args = ["./exec/nerd"]
	args.append("-train")
	args.append(f'data/processed/line/{dataset}.undirected.cites')
	args.append("-output1")
	args.append(outfile1)
	args.append("-output2")
	args.append(outfile2)
	args.append("-size")
	args.append("%d" % size)
	args.append("-binary")
	args.append("0")
	args.append("-negative")
	args.append("%d" % neg)
	args.append("-rho")
	args.append("%f" % rho)
	args.append("-threads")
	args.append("%f" % 1)
	args.append("-walkSize")
	args.append("%d" % 5)
	args.append("-samples")
	args.append("%d" % 100)
	call(args)

def grid_search(dataset):
	for neg in negs:
		for rho in rhos:
			for init in range(inits):
				directory = f'data/repeated-embedding/nerd/{init}'
				if not os.path.exists(directory):
					print(f'creating directory {directory}')
					os.mkdir(directory)
				outfile1 = f'{directory}/{dataset}.line_{neg}_{rho}.undirected.hub.emb'
				outfile2 = f'{directory}/{dataset}.line_{neg}_{rho}.undirected.aut.emb'
				run_nerd(dataset,outfile1,outfile2,neg,rho)


for dataset in ['cora_full']:
	grid_search(dataset)

