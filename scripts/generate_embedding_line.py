


from subprocess import call
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate embeddings for LINE (2nd-order proximity).")
parser.add_argument('--datasets',
                    nargs='+',
                    help='datasets to process, e.g., --dataset cora pubmed')


parser.add_argument('--hyper',
                    type = bool,
                    default=False,
                    help='Hyperparameter search? Default is False.')

inits = 20
negs = [3,5,8]
rhos = [0.005,0.025,0.125]

def run_line(dataset,directionality,outfile,order,neg,rho,size=128):
    if os.path.exists(outfile):
        print(f'{outfile} already exists!')
        return 
    args = ["./exec/line"]
    args.append("-train")
    args.append(f'data/graphs/line_format/{dataset}/{dataset}_{directionality}.cites')
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
    args.append("%f" % 4)
    args.append("-samples")
    args.append("%d" % 100)
    call(args)

def make_directory(directory):
    if not os.path.exists(directory):
        print(f'creating directory {directory}')
        os.mkdir(directory)

def run_line_multiple_inits(dataset,directionality,directory,order,neg,rho,inits):
    for init in range(inits):
        outfile = f'{directory}/{dataset}.line_neg{neg}_rho{rho}_init{init}.{directionality}.emb'
        run_line(dataset,directionality,outfile,order,neg,rho)

def grid_search(dataset,directionality,order):
    for neg in negs:
        for rho in rhos:
            directory = f'data/embedding/line{order}/{dataset}'
            make_directory(directory)
            run_line_multiple_inits(dataset,directionality,directory,order,neg,rho,inits)


def extract_hyperparams(df_hyper):
    print(df_hyper)
    return int(df_hyper.neg), df_hyper.lr

args = parser.parse_args()
directionalities = 'undirected directed reversed'.split()

if args.hyper:
    for dataset in args.datasets:
        for directionality in directionalities:
            grid_search(dataset,directionality,2)
else:
    for dataset in args.datasets:
        for directionality in directionalities:
            directory = f'data/embedding/line2/{dataset}'
            make_directory(directory)
            df = pd.read_csv('reports/results/eval/line2.csv')
            df = df[df.val_avg == df.val_avg.max()].reset_index().loc[0]
            neg, rho = extract_hyperparams(df)
            run_line_multiple_inits(dataset,directionality,directory,2,neg,rho,inits)
        

