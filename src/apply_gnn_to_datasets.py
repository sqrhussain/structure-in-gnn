from src.evaluation.gnn_evaluation_module import eval_gnn
from src.models.gat_models import MonoGAT#, BiGAT, TriGAT
from src.models.rgcn_models import MonoRGCN
from src.models.multi_layered_model import MonoModel#, BiModel, TriModel
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, RGCNConv
from src.data.data_loader import GraphDataset
import warnings
import pandas as pd
import os
import argparse
def parse_args():

    parser = argparse.ArgumentParser(description="Test accuracy for GCN/SAGE/GAT/RGCN")
    parser.add_argument('--size',
                        type=int,
                        default=96,
                        help='Channel size. Default is 12.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate. Default is 0.01.')
    parser.add_argument('--wd',
                        type=float,
                        default=0.01,
                        help='Regularization weight. Default is 0.01.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.8,
                        help='Dropout probability. Default is 0.6.')
    parser.add_argument('--conf',
                        type=bool,
                        default=False,
                        help='Is configuration model evaluation. Default is False.')
    parser.add_argument('--sbm',
                        type=bool,
                        default=False,
                        help='Is SBM evaluation. Default is False.')
    parser.add_argument('--flipped',
                        type=bool,
                        default=False,
                        help='Evaluating with flipped edges? Default is False.')
    parser.add_argument('--heads',
                        type=int,
                        default=4,
                        help='Attention heads. Default is 4.')
    parser.add_argument('--attention_dropout',
                        type=float,
                        default=0.4,
                        help='Attention dropout for GAT. Default is 0.4.')
    parser.add_argument('--dataset',
                        default="cora",
                        help='Dataset name. Default is cora.')
    parser.add_argument('--model',
                        default="gcn",
                        help='Model name. Default is GCN.')
    parser.add_argument('--splits',
                        type=int,
                        default=100,
                        help='Number of random train/validation/test splits. Default is 100.')
    parser.add_argument('--runs',
                        type=int,
                        default=20,
                        help='Number of random initializations of the model. Default is 20.')
    parser.add_argument('--conf_inits',
                        type=int,
                        default=10,
                        help='Number of configuration model runs. Default is 10.')
    parser.add_argument('--sbm_inits',
                        type=int,
                        default=10,
                        help='Number of SBM runs. Default is 10.')
    parser.add_argument('--directionality',
                        default='undirected',
                        help='Directionality: undirected/directed/reversed. Default is undirected.')
                        
    parser.add_argument('--train_examples',
                        type=int,
                        default=20,
                        help='Number of training examples per class. Default is 20.')
    parser.add_argument('--val_examples',
                        type=int,
                        default=30,
                        help='Number of validation examples per class. Default is 30.')
                        
    args = parser.parse_args()
    return args

name2conv = {'gcn': GCNConv, 'sage': SAGEConv, 'gat': GATConv, 'rgcn': RGCNConv}

def eval_archs_gat(dataset, channel_size, dropout, lr, wd, heads,attention_dropout,runs,splits,train_examples,val_examples, models=[MonoGAT],isDirected = False):
    if isDirected:
        models = [MonoGAT]
    return eval_gnn(dataset, GATConv, channel_size, dropout, lr, wd, heads=heads, attention_dropout=attention_dropout,
                      models=models, num_runs=runs, num_splits=splits, test_score=True,
                      train_examples = train_examples, val_examples = val_examples)


def eval_archs_gcn(dataset, conv, channel_size, dropout, lr, wd, runs,splits,train_examples,val_examples, models=[MonoModel], isDirected=False):
    if isDirected:
        models = [MonoModel]
    return eval_gnn(dataset, conv, channel_size, dropout, lr, wd, heads=1,attention_dropout=0.3, # dummy values for heads and attention_dropout
                      models=models, num_runs=runs, num_splits=splits,test_score=True,
                      train_examples = train_examples, val_examples = val_examples)



def eval_archs_rgcn(dataset, channel_size, dropout, lr, wd, runs,splits,train_examples,val_examples, models=[MonoRGCN]):
    return eval_gnn(dataset, RGCNConv, channel_size, dropout, lr, wd, heads=1,attention_dropout=0.3,  # dummy values for heads and attention_dropout
                      models=models, num_runs=runs, num_splits=splits,test_score=True,
                      train_examples = train_examples, val_examples = val_examples)



def eval(model, dataset, channel_size, dropout, lr, wd, heads, attention_dropout, runs, splits, train_examples, val_examples, isDirected):
    if model == 'gat':
        return eval_archs_gat(dataset, channel_size, dropout, lr, wd, heads, attention_dropout, splits=splits, runs=runs, train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
    elif model == 'rgcn':
        return eval_archs_rgcn(dataset, channel_size, dropout, lr, wd, splits=splits, runs=runs, train_examples = train_examples, val_examples = val_examples)
    else:
        return eval_archs_gcn(dataset, name2conv[model], channel_size, dropout, lr, wd, splits=splits, runs=runs, train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)

def eval_original(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}', dataset_name,
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.cites',
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                           directed=isDirected, reverse=isReversed)
    df_cur = eval(model=model, dataset=dataset, channel_size=size, lr=lr, splits=splits, runs=runs,
                  dropout=dropout, wd=wd, heads=heads, attention_dropout=attention_dropout,
                  train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
    return df_cur

def eval_random(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-random', dataset_name,
                           f'data/graphs/random/{dataset_name}/{dataset_name}.cites',
                           f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                           directed=isDirected, reverse=isReversed)
    df_cur = eval(model=model, dataset=dataset, channel_size=size, lr=lr, splits=splits, runs=runs,
                  dropout=dropout, wd=wd, heads=heads, attention_dropout=attention_dropout,
                  train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
    return df_cur

def eval_conf(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, conf_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(conf_inits):
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-confmodel{i}', dataset_name,
                               f'data/graphs/confmodel/{dataset_name}/{dataset_name}_confmodel_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)
        df_cur = eval(model=model, dataset=dataset, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['confmodel_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val

def eval_sbm(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, sbm_inits):
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in range(sbm_inits):
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-sbm{i}', dataset_name,
                               f'data/graphs/sbm/{dataset_name}/{dataset_name}_sbm_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)
        df_cur = eval(model=model, dataset=dataset, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['sbm_num'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val
    
def eval_flipped(model, dataset_name, directionality, size, dropout, lr, wd, heads,attention_dropout,
        splits, runs, train_examples, val_examples, percentages=range(10,51,10)):
    print(percentages)
    isDirected = (directionality != 'undirected')
    isReversed = (directionality == 'reversed')
    df_val = pd.DataFrame()
    for i in percentages:
        print(f'data/graphs/processed/{dataset_name}/{dataset_name}.content')
        dataset = GraphDataset(f'data/tmp/{dataset_name}{("_" + directionality) if isDirected else ""}-flipped{i}', dataset_name,
                               f'data/graphs/flip_edges/{dataset_name}/{dataset_name}_{i}.cites',
                               f'data/graphs/processed/{dataset_name}/{dataset_name}.content',
                               directed=isDirected, reverse=isReversed)
        df_cur = eval(model=model, dataset=dataset, channel_size=size, lr=lr, splits=splits, runs=runs,
                      dropout=dropout, wd=wd, heads=heads,attention_dropout=attention_dropout,
                      train_examples = train_examples, val_examples = val_examples,isDirected=isDirected)
        df_cur['percentage'] = i
        df_val = pd.concat([df_val, df_cur])
    return df_val
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_args()
    if args.directionality not in {'undirected', 'reversed', 'directed'}:
        print("--directionality must be in {'undirected','reversed','directed'}")
        exit(1)
    isDirected = (args.directionality != 'undirected')
    isReversed = (args.directionality == 'reversed')

    

    # TODO find a better way to create names
    val_out = f'reports/results/test_acc/{args.model}_{args.dataset}{"_conf" if args.conf else ""}' \
              f'{"_sbm" if args.sbm else ""}{("_" + args.directionality) if isDirected else ""}.csv'

    if os.path.exists(val_out):
        df_val = pd.read_csv(val_out)
    else:
        df_val = pd.DataFrame(
            columns='conv arch ch dropout lr wd heads attention_dropout splits inits val_accs val_avg val_std'
                    ' test_accs test_avg test_std stopped elapsed'.split())
    if args.conf:
        df_cur = eval_conf(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, args.conf_inits)
    elif args.sbm:
        df_cur = eval_sbm(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples, args.sbm_inits)
    elif args.flipped:
        df_cur = eval_flipped(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples)
    else:
        df_cur = eval_original(args.model, args.dataset, args.directionality, args.size, args.dropout, args.lr, args.wd,
                args.heads, args.attention_dropout,
                args.splits, args.runs, args.train_examples, args.val_examples)
    df_val = pd.concat([df_val, df_cur])
    df_val.to_csv(val_out, index=False)
