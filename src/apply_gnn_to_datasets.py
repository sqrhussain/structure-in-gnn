from src.evaluation.gnn_evaluation_module import eval_gnn
from src.models.gat_models import MonoGAT, BiGAT, TriGAT
from src.models.rgcn_models import MonoRGCN
from src.models.multi_layered_model import MonoModel, BiModel, TriModel
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, RGCNConv
from src.data.data_loader import GraphDataset
import warnings
import pandas as pd
import os
import argparse

warnings.filterwarnings('ignore')

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
parser.add_argument('--heads',
                    type=int,
                    default=4,
                    help='Attention heads. Default is 4.')
parser.add_argument('--dataset',
                    default="pubmed",
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
args = parser.parse_args()

if args.directionality not in {'undirected', 'reversed', 'directed'}:
    print("--directionality must be in {'undirected','reversed','directed'}")
    exit(1)


isDirected = (args.directionality != 'undirected')
isReversed = (args.directionality == 'reversed')
dataset = GraphDataset(f'data/tmp/{args.dataset}{("_" + args.directionality) if isDirected else ""}', args.dataset,
                       f'data/graphs/processed/{args.dataset}/{args.dataset}.cites',
                       f'data/graphs/processed/{args.dataset}/{args.dataset}.content',
                       directed=isDirected, reverse=isReversed)

name2conv = {'gcn': GCNConv, 'sage': SAGEConv, 'gat': GATConv, 'rgcn': RGCNConv}

# TODO find a better way to create names
val_out = f'reports/results/test_acc/{args.model}_{args.dataset}{"_conf" if args.conf else ""}' \
          f'{"_sbm" if args.sbm else ""}{("_" + args.directionality) if isDirected else ""}.csv'

if os.path.exists(val_out):
    df_val = pd.read_csv(val_out)
else:
    df_val = pd.DataFrame(
        columns='conv arch ch dropout lr wd heads splits inits val_accs val_avg val_'
                'std test_accs test_avg test_std stopped elapsed'.split())


def eval_archs_gat(dataset, channel_size, dropout, lr, wd, heads, models=[MonoGAT, BiGAT, TriGAT]):
    if isDirected:
        models = [MonoGAT]
    return eval_gnn(dataset, GATConv, channel_size, dropout, lr, wd, heads=heads,
                      models=models, num_runs=args.runs, num_splits=args.splits, test_score=True)


def eval_archs_gcn(dataset, conv, channel_size, dropout, lr, wd, models=[MonoModel, BiModel, TriModel]):
    if isDirected:
        models = [MonoModel]
    return eval_gnn(dataset, conv, channel_size, dropout, lr, wd, heads=1,
                      models=models, num_runs=args.runs, num_splits=args.splits,test_score=True)


def eval_archs_rgcn(dataset, channel_size, dropout, lr, wd, models=[MonoRGCN]):
    return eval_gnn(dataset, RGCNConv, channel_size, dropout, lr, wd, heads=1,
                      models=models, num_runs=args.runs, num_splits=args.splits,test_score=True)


def eval(dataset, channel_size, dropout, lr, wd, heads):
    if args.model == 'gat':
        return eval_archs_gat(dataset, channel_size, dropout, lr, wd, heads)
    elif args.model == 'rgcn':
        return eval_archs_rgcn(dataset, channel_size, dropout, lr, wd)
    else:
        return eval_archs_gcn(dataset, name2conv[args.model], channel_size, dropout, lr, wd)


def eval_conf_model(df):
    df["confmodel_num"] = ""
    for i in range(args.conf_inits):
        dataset = GraphDataset(f'data/tmp/{args.dataset}-confmodel{i}', args.dataset,
                               f'data/graphs/confmodel/{args.dataset}/{args.dataset}_confmodel_{i}.cites',
                               f'data/graphs/processed/{args.dataset}/{args.dataset}.content',
                               directed=isDirected, reverse=isReversed)
        df_cur = eval(dataset=dataset, channel_size=args.size, lr=args.lr,
                      dropout=args.dropout, wd=args.wd, heads=args.heads)
        df_cur['confmodel_num'] = i
        df = pd.concat([df, df_cur])
        df.to_csv(val_out, index=False)


if args.conf:
    eval_conf_model(df_val)
else:
    df_cur = eval(dataset=dataset, channel_size=args.size, lr=args.lr,
                  dropout=args.dropout, wd=args.wd, heads=args.heads)

    df_val = pd.concat([df_val, df_cur])
    df_val.to_csv(val_out, index=False)
