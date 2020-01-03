from gnn_hyperparam_search import eval_archs
from gat_models import MonoGAT, BiGAT, TriGAT
from torch_geometric.nn import GATConv
from cora_loader import CitationNetwork
import warnings
import pandas as pd
import os
import argparse
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description = "Hyperparameter search for GAT")

parser.add_argument('--size',
                    type = int,
                    default = 96,
                help = 'Channel size. Default is 96.')

parser.add_argument('--lr',
                    type = float,
                    default = 0.001,
                help = 'Learning rate. Default is 0.001.')

parser.add_argument('--directionality',
                default = "undirected",
                help = 'Model name. Default is undirected.')
args = parser.parse_args()



cora = CitationNetwork(f'/tmp/cora3{args.directionality}','cora',
    directed=(args.directionality != "undirected"),
    reverse=(args.directionality == "reversed"))
num_splits = 100
num_runs = 20
sz = args.size
lrs = [1e-3,5e-3,1e-2] # args.lr #

val_out = f'eval/gat_val_cora_{args.directionality}.csv'

if os.path.exists(val_out):
    df_val = pd.read_csv(val_out)
else:
    df_val = pd.DataFrame(columns='conv arch ch dropout lr wd heads splits inits val_accs val_avg val_std test_accs test_avg test_std stopped elapsed'.split())
        
def eval_archs_gat(dataset,channel_size,dropout,lr,wd,heads,models=[MonoGAT, BiGAT, TriGAT],df=None):
    return eval_archs(dataset,GATConv,channel_size,dropout,lr,wd,heads=heads,
               models=models,num_runs=num_runs,num_splits=num_splits,df_val = df)

def contains(df_val,ch,lr,dropout,wd,heads):
    return ((df_val['ch']==ch) & (df_val['lr']==lr) & (df_val['dropout']==dropout) & (df_val['wd']==wd) & (df_val['heads']==heads)).any()

for conv in [GATConv]:
    for ch in [sz]:
        for lr in lrs:
            for dropout in [0.2,0.4,0.6,0.8]:
                for wd in [1e-4,1e-3,1e-2,1e-1]:
                    for heads in [4]:   # flaw when ch==12 and heads==4 and model is BiGAT (12//2//4 => 1)
                                            # resulting in smaller representation size than expected (ch -> 8 not 12)
                        if contains(df_val,ch,lr,dropout,wd,heads):
                            print('already calculated!')
                            continue
                        df_val = eval_archs_gat(dataset=cora,channel_size=ch,lr=lr,
                                                dropout=dropout,wd=wd,heads=heads,df=df_val)
                        df_val.to_csv(val_out,index=False)