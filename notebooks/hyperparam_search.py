from gnn_hyperparam_search import eval_archs
from multi_layered_model import MonoModel,BiModel,TriModel
from torch_geometric.nn import GCNConv,SAGEConv
from cora_loader import CitationNetwork
import warnings
import pandas as pd 
import os
import argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description = "Hyperparameter search for GCN/SAGE")


parser.add_argument('--model',
                default = "gcn",
                help = 'Model name. Default is GCN.')


parser.add_argument('--directionality',
                default = "undirected",
                help = 'Model name. Default is GCN.')
name2conv = {'gcn':GCNConv, 'sage':SAGEConv}
args = parser.parse_args()

cora = CitationNetwork(f'/tmp/cora3{args.directionality}','cora',
    directed=(args.directionality != "undirected"),
    reverse=(args.directionality == "reversed"))

num_splits = 100
num_runs = 20

val_out = f'eval/{args.model}_val_cora_{args.directionality}.csv'

if os.path.exists(val_out):
    df_val = pd.read_csv(val_out)
else:
    df_val = pd.DataFrame(columns='conv arch ch dropout lr wd heads splits inits val_accs val_avg val_std test_accs test_avg test_std stopped elapsed'.split())
    
def eval_archs_gcn(dataset,conv,channel_size,dropout,lr,wd,models=[MonoModel, BiModel, TriModel],df=None):
    return eval_archs(dataset,conv,channel_size,dropout,lr,wd,heads=1,
               models=models,num_runs=num_runs,num_splits=num_splits,df_val = df)

def contains(df_val,ch,lr,dropout,wd):
    return ((df_val['ch']==ch) & (df_val['lr']==lr) & (df_val['dropout']==dropout) & (df_val['wd']==wd)).any()

for ch in [96]:
    for lr in [1e-3,5e-3,1e-2]:
        for dropout in [0.2,0.4,0.6,0.8]:
            for wd in [1e-4,1e-3,1e-2,1e-1]:
                if contains(df_val,ch,lr,dropout,wd):
                    print('already calculated!')
                    continue
                df_val = eval_archs_gcn(dataset=cora,conv=name2conv[args.model],channel_size=ch,lr=lr,dropout=dropout,wd=wd,df=df_val)
                df_val.to_csv(val_out,index=False)