from src.apply_gnn_to_datasets import eval_original, eval_conf, eval_sbm
import argparse
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Test accuracy for GCN/SAGE/GAT/RGCN")

    parser.add_argument('--conf',
                        type=bool,
                        default=False,
                        help='Is configuration model evaluation. Default is False.')
    parser.add_argument('--sbm',
                        type=bool,
                        default=False,
                        help='Is SBM evaluation. Default is False.')

    parser.add_argument('--datasets',
                        nargs='+',
                        help='datasets to process, e.g., --dataset cora pubmed')
    parser.add_argument('--models',
                        nargs='+',
                        help='models to evaluate, e.g., --models gcn sage gat')
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

    args = parser.parse_args()
    return args

def model_selection(model, dataset):
    if dataset == 'cora_full':
        dataset = 'pubmed' # take pubmed hyperparams and apply them to cora_full
    # if dataset == 'citeseer':
    #     dataset = 'cora' # take cora hyperparams and apply them to citeseer
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
    attdrop = 0.3
    if 'attention_dropout' in df_hyper:
        attdrop = df_hyper.attention_dropout
    return int(df_hyper.ch), df_hyper.dropout, df_hyper.lr, df_hyper.wd, int(df_hyper.heads), attdrop

if __name__ == '__main__':
    args = parse_args()
    print(f'datasets: {args.datasets}')
    print(f'models: {args.models}')
    df_hyper = pd.DataFrame()
    for model in args.models:
        for dataset in args.datasets:
            df_hyper = df_hyper.append(model_selection(model,dataset))
    print(df_hyper)
    for directionality in ['undirected', 'directed', 'reversed']:
        for model in args.models:
            for dataset in args.datasets:
                print(f'{dataset}, {model}, {directionality}')
                val_out = f'reports/results/test_acc/{model}_{dataset}{"_conf" if args.conf else ""}' \
                          f'{"_sbm" if args.sbm else ""}{("_" + directionality) if (directionality!="undirected") else ""}.csv'
                if os.path.exists(val_out):
                    df_val = pd.read_csv(val_out)
                else:
                    df_val = pd.DataFrame(
                        columns='conv arch ch dropout lr wd heads attention_dropout splits inits val_accs val_avg val_std'
                                ' test_accs test_avg test_std stopped elapsed'.split())
                size, dropout, lr, wd, heads, attention_dropout = extract_hyperparams(df_hyper,dataset,model)
                if args.sbm:
                    df_cur = eval_sbm(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                elif args.conf:
                    df_cur = eval_conf(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                else:
                    df_cur = eval_original(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples)
                df_val = pd.concat([df_val,df_cur])
                df_val.to_csv(val_out, index=False)
