from src.apply_gnn_to_datasets import eval_archs_gcn
from torch_geometric.nn import GCNConv
import pandas as pd


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

dataset = 'cora'
model = 'gcn'

df_hyper = pd.DataFrame()
df_hyper = df_hyper.append(model_selection(model,dataset))
size, dropout, lr, wd, heads = extract_hyperparams(df_hyper,dataset,model)

res = eval_archs_gcn(dataset, GCNConv, size, dropout, lr, wd, splits=10, runs=5)

print(res)