from embedding_evaluation_framework import EmbeddingData, eval_n2v
import pandas as pd
import numpy as np
import os

lrs = [0.005,0.025,0.125]
negs = [5,10,15,20]


inits = 20
splits = 100

val_out = 'eval/line-2.csv'


if os.path.exists(val_out):
    df_n2v = pd.read_csv(val_out)
else:
    df_n2v = pd.DataFrame(columns='splits inits lr neg val_acc val_avg val_std'.split())


def contains(df_val,lr,neg):
    return ((df_val['lr']==lr) & (df_val['neg']==neg)).any()

for lr in lrs:
    for neg in negs:
        if contains(df_n2v,lr,neg):
            print('already calculated!')
            continue
        print(f'evaluating lr={lr},neg={neg}')
        vals = []
        tests = []
        for init in range(inits):
            print(f' init {init}')
            emb = EmbeddingData(f'/tmp/line2Undlr{lr}neg{neg}EmbCoraInit{init}','cora',f'line_{neg}_{lr}',directed=False,initialization=f'line2/{init}')
            val = eval_n2v(emb[0],num_splits=splits)
            vals = vals + val
#             tests = tests + test
        df_n2v = df_n2v.append({'splits':splits, 'inits':inits, 'lr':lr, 'neg':neg,
                                'val_acc':vals, 'val_avg':np.mean(vals), 'val_std':np.std(vals),
#                                 'test_acc':tests, 'test_avg':np.mean(tests), 'test_std':np.std(tests)
                               },ignore_index=True)
        df_n2v.to_csv(val_out,index=False)