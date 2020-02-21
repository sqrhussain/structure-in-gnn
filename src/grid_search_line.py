


from subprocess import call
import os

inits = 20
negs = [5]
rhos = [0.005]

def run_line(dataset,outfile,order,neg,rho,size=128):
	if os.path.exists(outfile):
		print(f'{outfile} already exists!')
		return 
	args = ["./exec/line"]
	args.append("-train")
	args.append(f'data/processed/line/{dataset}.undirected.cites')
	args.append("-output")
	args.append(outfile)
	args.append("-size")
	args.append("%d" % size)
	args.append("-binary")
	args.append("0")
	args.append("-order")
	args.append("%d" % order)
	args.append("-negative")
	args.append("%d" % neg)
	args.append("-rho")
	args.append("%f" % rho)
	args.append("-threads")
	args.append("%f" % 1)
	args.append("-samples")
	args.append("%d" % 100)
	call(args)

def grid_search(dataset,order):
	for neg in negs:
		for rho in rhos:
			for init in range(inits):
				directory = f'data/repeated-embedding/line{order}/{init}'
				if not os.path.exists(directory):
					print(f'creating directory {directory}')
					os.mkdir(directory)
				outfile = f'{directory}/{dataset}.line_{neg}_{rho}.undirected.emb'
				run_line(dataset,outfile,order,neg,rho)


for dataset in ['cora_full']:
	grid_search(dataset,1)
	grid_search(dataset,2)

