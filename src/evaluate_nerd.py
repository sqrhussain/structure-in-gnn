from embedding_evaluation_framework import EmbeddingData, eval_n2v
import pandas as pd
import numpy as np
import os

lrs = [0.005,0.025,0.125]
negs = [5,10,15,20]

inits = 20
splits = 100

val_out = 'eval/nerd.csv'

if os.path.exists(val_out):
    df_nerd = pd.read_csv(val_out)
else:
    df_nerd = pd.DataFrame(columns='splits inits type lr neg val_acc val_avg val_std'.split())

def contains(df_val,tp,lr,neg):
    return ((df_val['type']==tp) & (df_val['lr']==lr) & (df_val['neg']==neg)).any()


for neg in negs:
    for lr in lrs:
        for tp in ['hub','aut']:
            if contains(df_nerd,tp,lr,neg):
                print('already calculated!')
                continue
            print(f'evaluating type={tp} lr={lr}, neg={neg}')
            vals = []
            tests = []
            for init in range(inits):
                print(f' init {init}')
                target_path = f'../data/repeated-embedding/nerd/{init}/cora.line_{neg}_{lr}.undirected.{tp}.emb'
                if not os.path.exists(target_path):
                    print(f'{target_path} does not exist')
                emb = EmbeddingData(f'/tmp/nerdUndlr{lr}neg{neg}EmbCoraInit{init}','cora',f'line_{neg}_{lr}',directed=False,initialization=f'nerd/{init}',nerd='aut')
                val = eval_n2v(emb[0],num_splits=splits)
                vals = vals + val
    #             tests = tests + test
            df_nerd = df_nerd.append({'splits':splits, 'inits':inits, 'type':tp, 'lr':lr, 'neg':neg,
                                    'val_acc':vals, 'val_avg':np.mean(vals), 'val_std':np.std(vals),
    #                                 'test_acc':tests, 'test_avg':np.mean(tests), 'test_std':np.std(tests)
                                   },ignore_index=True)
            df_nerd.to_csv(val_out,index=False)