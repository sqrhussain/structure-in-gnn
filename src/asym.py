from src.apply_gnn_to_datasets import eval_archs_gcn,eval_archs_gat
from torch_geometric.nn import GCNConv,SAGEConv
import pandas as pd
import os
from src.data.data_loader import GraphDataset
from src.models.multi_layered_model import ASYM,pASYM,MonoModel
import warnings
warnings.filterwarnings("ignore")

def model_selection(model, dataset):
    if dataset == 'cora_full':
        dataset = 'pubmed' # take pubmed hyperparams and apply them to cora_full
    filename = f'reports/results/eval/{model}_val_{dataset}_undirected.csv'
    if not os.path.exists(filename):
        filename = f'reports/results/eval/{model}.csv'
    df = pd.read_csv(filename)
    df = df[df.val_avg == df.val_avg.max()].reset_index().loc[0]
    df['dataset'] = dataset
    df['conv'] = model
    return df

def extract_hyperparams(df_hyper, dataset, model):
    df_hyper = df_hyper[(df_hyper.dataset == dataset) & (df_hyper.conv == model)].reset_index().loc[0]
    return int(df_hyper.ch), df_hyper.dropout, df_hyper.lr, df_hyper.wd, int(df_hyper.heads)

dataset = 'pubmed'
model = 'gcn'
directionality = 'undirected'
isDirected = (directionality!='undirected')
isReversed = (directionality=='reversed')

df_hyper = pd.DataFrame()
df_hyper = df_hyper.append(model_selection(model,dataset))
size, dropout, lr, wd, heads = extract_hyperparams(df_hyper,dataset,model)

data = GraphDataset(f'data/tmp/{dataset}{("_" + directionality) if isDirected else ""}', dataset,
                       f'data/graphs/processed/{dataset}/{dataset}.cites',
                       f'data/graphs/processed/{dataset}/{dataset}.content',
                       directed=isDirected, reverse=isReversed)
res = eval_archs_gcn(data, SAGEConv, size, dropout, lr, wd, splits=100, runs=20, train_examples=20, val_examples=30,models=[ASYM,pASYM])
# res = eval_archs_gat(data, size, dropout, lr, wd, heads, splits=100, runs=20, train_examples=20, val_examples=30,models=[ASYM,pASYM])

res.to_csv(f'reports/results/test_acc/asym_{model}_{dataset}.csv')
