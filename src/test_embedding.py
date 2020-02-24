import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from evaluation.embedding_evaluation_framework import test_n2v
from data.cora_loader import EmbeddingData

import argparse

parser = argparse.ArgumentParser(description = "Test accuracy for repeated embeddings.\n" +
                                    " Iterates over (repeated-embedding/<dataset>/<i>/<filename>.emb) and evaluates them,\n"+
                                    " where <i> is in [0..19]\n" + 
                                    "")
                                    # " and <dataset> is in ['cora','citeseer','PubMed','cora_full'].\n")
parser.add_argument('--dataset',
                default = "cora",
                help = 'Dataset name. Default "cora".')
parser.add_argument('--method',
                help = 'method name. Example "node2vec"')
parser.add_argument('--filename',
                help = 'File name in the embedding folder. Example "node2vec_rw_0.75_1.75"')
parser.add_argument('--directed',
                type=bool,
                default=False,
                help = 'Is directed? Default False.')
parser.add_argument('--reversed',
                type=bool,
                default=False,
                help = 'Is reversed? Default False.')
parser.add_argument('--nerd',
                default='aut',
                help = 'NERD method embedding (aut/hub). Default "aut".')
parser.add_argument('--splits',
                type = int,
                default = 100,
                help = 'Number of random train/validation/test splits. Default is 100.')
parser.add_argument('--runs',
                type = int,
                default = 20,
                help = 'Number of random initializations of the model. Default is 20.')
args = parser.parse_args()


isDirected = (args.directed or args.reversed)
isReversed = args.reversed
directionality = ("reversed" if isReversed else ("directed" if isDirected else "undirected"))
nerdStatus = (args.nerd if args.method == "nerd" else None)
nerdString = (('_' + nerdStatus) if nerdStatus is not None else "")
inits = args.runs
num_splits = args.splits

def report_test_acc_unsupervised_embedding(cache_prefix,dataset,filename,method,speedup=False):
    tests = []
    for init in range(inits):
        print(f'{dataset} init {init}\n')
        emb = EmbeddingData(f'/tmp/{args.method}{nerdString}_{dataset}_{directionality}_{cache_prefix}EmbInit{init}',
            dataset,filename,directed=isDirected,reverse=isReversed,initialization=f'{method}/{init}',
            nerd = nerdStatus)
        print(f'started test {init}')
        test = test_n2v(emb,num_splits=num_splits,speedup=speedup)
        print(str(test) + '\n')
        tests = tests + test
    return tests

# datasets = ['cora','citeseer','PubMed','cora_full']

if args.dataset == '.all':
    datasets = ['cora','citeseer','PubMed','cora_full']
else:
    datasets = [args.dataset]

def model_selection_gnn(df):
    return df[df.val_avg == df.groupby('arch').val_avg.transform(max)]
def model_selection(df):
    return df[df.val_avg == df.val_avg.max()]


test_out = f'test_acc/emb_eval_{args.method}{nerdString}_{args.dataset}_{directionality}.csv'

if os.path.exists(test_out):
    test_acc = pd.read_csv(test_out)
else:
    test_acc = pd.DataFrame(columns='method dataset test_acc test_avg test_std'.split())


for dataset in datasets:
    tests = report_test_acc_unsupervised_embedding(cache_prefix=args.filename,dataset=dataset,filename=args.filename,
                                                   method=args.method,speedup=(dataset=='cora_full'))
    test_acc = test_acc.append({'method':args.method, 'dataset':dataset,
                    'test_acc':tests, 'test_avg':np.mean(tests), 'test_std':np.std(tests)},ignore_index=True)
    test_acc.to_csv(test_out,index = False)
