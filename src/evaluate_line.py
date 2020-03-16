from subprocess import call
import os
import pandas as pd
import argparse
from src.evaluation.embedding_evaluation_module import report_test_acc_unsupervised_embedding
import numpy as np
parser = argparse.ArgumentParser(description="Evaluate embeddings for LINE (2nd-order proximity).")
parser.add_argument('--datasets',
                    nargs='+',
                    help='datasets to process, e.g., --dataset cora pubmed')

# parser.add_argument('--hyper',
#                     type = bool,
#                     default=False,
#                     help='Hyperparameter search? Default is False.')

num_inits = 20
num_splits = 100

def extract_hyperparams(df_hyper):
    print(df_hyper)
    return int(df_hyper.neg), df_hyper.lr


def train_count(dataset):
    if 'webkb' in dataset:
        return 10
    return 20


def val_count(dataset):
    if 'webkb' in dataset:
        return 15
    return 30


args = parser.parse_args()
directionalities = 'undirected directed reversed'.split()
method = 'line2'

val_out = f'reports/results/emb_acc/{method}.csv'

if os.path.exists(val_out):
    df_val = pd.read_csv(val_out)
else:
    df_val = pd.DataFrame(
        columns='method dataset directionality neg lr inits splits test_acc test_avg test_std'.split())

for dataset in args.datasets:
    for directionality in directionalities:
        directory = f'data/embedding/line2/{dataset}'
        df = pd.read_csv('reports/results/eval/line2.csv')
        df = df[df.val_avg == df.val_avg.max()].reset_index().loc[0]
        neg, rho = extract_hyperparams(df)

        acc = []
        for init in range(num_inits):
            embfile = f'{dataset}.line_neg{neg}_rho{rho}_init{init}.{directionality}.emb'
            embpath = f'{directory}/{embfile}'
            if not os.path.exists(embpath):
            	continue
            attrfile = f'data/graphs/processed/{dataset}/{dataset}.content'
            cur_acc = report_test_acc_unsupervised_embedding(f'data/tmp/line2{embfile}',dataset,embpath,attrfile,
                        num_splits,train_count(dataset), val_count(dataset))

            acc = acc + cur_acc
        row = {'method':method, 'dataset':dataset, 'directionality':directionality,
           'neg':neg, 'lr':rho, 'inits':num_inits, 'splits':num_splits,
           'test_acc':acc, 'test_avg':np.array(acc).mean(), 'test_std':np.array(acc).std()}
        df_val = df_val.append(row, ignore_index=True)
        df_val.to_csv(val_out, index=False) 
