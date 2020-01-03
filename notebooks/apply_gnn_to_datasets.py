from gnn_hyperparam_search import eval_archs
from gat_models import MonoGAT, BiGAT, TriGAT
from rgcn_models import MonoRGCN
from multi_layered_model import MonoModel,BiModel,TriModel
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,RGCNConv
from cora_loader import CitationNetwork
import warnings
import pandas as pd
import os
import argparse
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description = "Test accuracy for GCN/SAGE/GAT")
parser.add_argument('--size',
                type = int,
                default = 96,
                help = 'Channel size. Default is 12.')
parser.add_argument('--lr',
                type = float,
                default = 0.01,
                help = 'Learning rate. Default is 0.01.')
parser.add_argument('--wd',
                type = float,
                default = 0.01,
                help = 'Regularization weight. Default is 0.01.')
parser.add_argument('--dropout',
                type = float,
                default = 0.8,
                help = 'Dropout probability. Default is 0.6.')
parser.add_argument('--heads',
                type = int,
                default = 4,
                help = 'Attention heads. Default is 4.')
parser.add_argument('--dataset',
                default = "cora",
                help = 'Dataset name. Default is cora.')
parser.add_argument('--model',
                default = "gcn",
                help = 'Model name. Default is GCN.')
parser.add_argument('--splits',
                type = int,
                default = 100,
                help = 'Number of random train/validation/test splits. Default is 100.')
parser.add_argument('--runs',
                type = int,
                default = 20,
                help = 'Number of random initializations of the model. Default is 20.')
args = parser.parse_args()

dataset = CitationNetwork(f'/tmp/{args.dataset}3',args.dataset,directed=False)


name2conv = {'gcn':GCNConv, 'sage':SAGEConv,'gat':GATConv, 'rgcn':RGCNConv}


val_out = f'test_acc/{args.model}_{args.dataset}.csv'

if os.path.exists(val_out):
    df_val = pd.read_csv(val_out)
else:
    df_val = pd.DataFrame(columns='conv arch ch dropout lr wd heads splits inits val_accs val_avg val_std test_accs test_avg test_std stopped elapsed'.split())
        
def eval_archs_gat(dataset,channel_size,dropout,lr,wd,heads,models=[MonoGAT, BiGAT, TriGAT],df=None):
    return eval_archs(dataset,GATConv,channel_size,dropout,lr,wd,heads=heads,
               models=models,num_runs=args.runs,num_splits=args.splits,df_val = df)

# def contains_gat(df_val,ch,lr,dropout,wd,heads):
#     return ((df_val['ch']==ch) & (df_val['lr']==lr) & (df_val['dropout']==dropout) & (df_val['wd']==wd) & (df_val['heads']==heads)).any()

def eval_archs_gcn(dataset,conv,channel_size,dropout,lr,wd,models=[MonoModel, BiModel, TriModel],df=None):
    return eval_archs(dataset,conv,channel_size,dropout,lr,wd,heads=1,
               models=models,num_runs=args.runs,num_splits=args.splits,df_val = df)

def eval_archs_rgcn(dataset,channel_size,dropout,lr,wd,models=[MonoRGCN],df=None):
    return eval_archs(dataset,RGCNConv,channel_size,dropout,lr,wd,heads=1,
               models=models,num_runs=args.runs,num_splits=args.splits,df_val = df)

# def contains_gcn(df_val,ch,lr,dropout,wd):
#     return ((df_val['ch']==ch) & (df_val['lr']==lr) & (df_val['dropout']==dropout) & (df_val['wd']==wd)).any()

def eval(dataset,channel_size,dropout,lr,wd,heads,df):
    if args.model == 'gat':
        return eval_archs_gat(dataset,channel_size,dropout,lr,wd,heads,df=df)
    elif args.model == 'rgcn':
        return eval_archs_rgcn(dataset,channel_size,dropout,lr,wd,df=df)
    else:
        return eval_archs_gcn(dataset,name2conv[args.model],channel_size,dropout,lr,wd,df=df)

# def contains(df_val,ch,lr,dropout,wd,heads):
#     if args.model == 'gat':
#         return contains_gat(df_val,ch,lr,dropout,wd,heads)
#     else:
#         return contains_gcn(df_val,ch,lr,dropout,wd)

# if contains(df_val,args.size,args.lr,args.dropout,args.wd,args.heads):
#     print('already calculated!')
# else
df_val = eval(dataset=dataset,channel_size=args.size,lr=args.lr,
                dropout=args.dropout,wd=args.wd,heads=args.heads,df=df_val)
df_val.to_csv(val_out,index=False)
