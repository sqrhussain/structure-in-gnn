from src.evaluation.gnn_evaluation_module import eval_gnn
from src.models.multi_layered_model import MonoModel,BiModel,TriModel
from src.models.gat_models import MonoGAT, BiGAT, TriGAT
from torch_geometric.nn import GCNConv,SAGEConv,GATConv
from src.data.data_loader import GraphDataset
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

parser.add_argument('--dataset',
                default = "cora",
                help = 'Dataset name. Default is cora.')

parser.add_argument('--splits',
                    type=int,
                    default=100,
                    help='Number of random train/validation/test splits. Default is 100.')
parser.add_argument('--runs',
                    type=int,
                    default=20,
                    help='Number of random initializations of the model. Default is 20.')
parser.add_argument('--train_examples',
                    type=int,
                    default=20,
                    help='Number of training examples per class. Default is 20.')
parser.add_argument('--val_examples',
                    type=int,
                    default=30,
                    help='Number of validation examples per class. Default is 30.')

name2conv = {'gcn':GCNConv, 'sage':SAGEConv, 'gat':GATConv}
args = parser.parse_args()

isDirected = (args.directionality != 'undirected')

dataset = GraphDataset(f'data/tmp/{args.dataset}{("_" + args.directionality) if isDirected else ""}', args.dataset,
    f'data/graphs/processed/{args.dataset}/{args.dataset}.cites',
    f'data/graphs/processed/{args.dataset}/{args.dataset}.content',
    directed=(args.directionality != "undirected"),
    reverse=(args.directionality == "reversed"))
 
num_splits = args.splits
num_runs = args.runs

val_out = f'reports/results/eval/{args.model}_val_{args.dataset}_{args.directionality}.csv'

if os.path.exists(val_out):
    df_val = pd.read_csv(val_out)
else:
    df_val = pd.DataFrame(columns='conv arch ch dropout lr wd heads splits inits val_accs val_avg val_std test_accs test_avg test_std stopped elapsed'.split())
    
def eval_archs_gcn(dataset,conv,channel_size,dropout,lr,wd,models=[MonoModel, BiModel, TriModel]):
    if isDirected:
        models = [MonoModel]
    return eval_gnn(dataset,conv,channel_size,dropout,lr,wd,heads=1,
           models=models,num_runs=num_runs,num_splits=num_splits,
           train_examples = args.train_examples, val_examples = args.val_examples)

def eval_archs_gat(dataset, channel_size, dropout, lr, wd, heads, models=[MonoGAT, BiGAT, TriGAT]):
    if isDirected:
        models = [MonoGAT]
    return eval_gnn(dataset, GATConv, channel_size, dropout, lr, wd, heads=heads,
                      models=models, num_runs=args.runs, num_splits=args.splits,
                      train_examples = args.train_examples, val_examples = args.val_examples)
def contains(df_val,ch,lr,dropout,wd):
    return ((df_val['ch']==ch) & (df_val['lr']==lr) & (df_val['dropout']==dropout) & (df_val['wd']==wd)).any()
def contains_gat(df_val,ch,lr,dropout,wd,heads):
    return ((df_val['ch']==ch) & (df_val['lr']==lr) & (df_val['dropout']==dropout)  & (df_val['wd']==wd) & (df_val['heads']==heads)).any()

for ch in [96]:
    for lr in [1e2,5e-2,1e-1]:
        for dropout in [0.2]:
            for wd in [1e-2,1e-1,5e-1,1]:
                if args.model == 'gat':
                    for heads in [1,2,4]:
                        if contains_gat(df_val,ch,lr,dropout,wd,heads):
                            print('already calculated!')
                            continue
                        df_cur = eval_archs_gat(dataset=dataset,channel_size=ch,lr=lr,dropout=dropout,wd=wd,heads=heads)
                        df_val = pd.concat([df_val,df_cur])
                        df_val.to_csv(val_out,index=False)
                else:                        
                    if contains(df_val,ch,lr,dropout,wd):
                        print('already calculated!')
                        continue
                    df_cur = eval_archs_gcn(dataset=dataset,conv=name2conv[args.model],channel_size=ch,lr=lr,dropout=dropout,wd=wd)
                    df_val = pd.concat([df_val,df_cur])
                    df_val.to_csv(val_out,index=False)
