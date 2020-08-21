from src.apply_gnn_to_datasets import eval_original, eval_conf, eval_sbm
from src.apply_gnn_to_datasets import eval_random, eval_erdos, eval_flipped, eval_removed_hubs, eval_added_2hop_edges
from src.apply_gnn_to_datasets import eval_label_sbm, eval_injected_edges, eval_injected_edges_sbm, eval_injected_edges_constant_nodes
from src.apply_gnn_to_datasets import eval_sbm_swap, eval_injected_edges_degree_cat,eval_injected_edges_attack_target
import argparse
import pandas as pd
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Main GNN training script.")

    parser.add_argument('--conf',
                        type=bool,
                        default=False,
                        help='Is configuration model evaluation. Default is False.')
    parser.add_argument('--sbm',
                        type=bool,
                        default=False,
                        help='Is SBM evaluation. Default is False.')
    parser.add_argument('--sbm_swap',
                        type=bool,
                        default=False,
                        help='Is SBM evaluation with synthetic swapping. Default is False.')
    parser.add_argument('--random',
                        type=bool,
                        default=False,
                        help='Evaluating on a completely random d-regular graph?. Default is False.')
    parser.add_argument('--erdos',
                        type=bool,
                        default=False,
                        help='Evaluating on a completely random Erdos-Renyi graph?. Default is False.')
    parser.add_argument('--label_sbm',
                        type=bool,
                        default=False,
                        help='Evaluating on a completely label_sbm graph?. Default is False.')
    parser.add_argument('--injected_edges',
                        type=bool,
                        default=False,
                        help='Evaluating on a completely injected_edges graph?. Default is False.')
    parser.add_argument('--injected_edges_degree_cat',
                        type=bool,
                        default=False,
                        help='Evaluating on a completely injected_edges_degree_cat graph?. Default is False.')
    parser.add_argument('--injected_edges_constant_nodes',
                        type=bool,
                        default=False,
                        help='Evaluating on a completely injected_edges_constant_nodes graph?. Default is False.')
    parser.add_argument('--injected_edges_attack_target',
                        type=bool,
                        default=False,
                        help='Evaluating on a completely injected_edges_attack_target graph?. Default is False.')
    parser.add_argument('--injected_edges_sbm',
                        type=bool,
                        default=False,
                        help='Evaluating on a completely injected_edges_sbm graph?. Default is False.')
    parser.add_argument('--flipped',
                        type=bool,
                        default=False,
                        help='Evaluating with flipped edges? Default is False.')
    parser.add_argument('--removed_hubs',
                        type=bool,
                        default=False,
                        help='Evaluating with removed hubs? Default is False.')
    parser.add_argument('--added_2hop_edges',
                        type=bool,
                        default=False,
                        help='Evaluating with added 2-hop edges? Default is False.')
                        
    parser.add_argument('--hubs_experiment',
                        default='weak',
                        help='hubs experiment type (loca_hubs, global_hubs, local_edges, global_edges). Default is None.')

    parser.add_argument('--datasets',
                        nargs='+',
                        help='datasets to process, e.g., --dataset cora pubmed')
    parser.add_argument('--models',
                        nargs='+',
                        help='models to evaluate, e.g., --models gcn sage gat')
    parser.add_argument('--splits',
                        type=int,
                        default=20,
                        help='Number of random train/validation/test splits. Default is 20.')
    parser.add_argument('--runs',
                        type=int,
                        default=5,
                        help='Number of random initializations of the model. Default is 5.')

    parser.add_argument('--train_examples',
                        type=int,
                        default=20,
                        help='Number of training examples per class. Default is 20.')
    parser.add_argument('--val_examples',
                        type=int,
                        default=30,
                        help='Number of validation examples per class. Default is 30.')
    parser.add_argument('--directionalities',
                        nargs='+',
                        default=['undirected'],
                        help='directionalities, example: --directionalities undirected directed reversed. Default: undirected')

    args = parser.parse_args()
    return args

datasets_similar_to_pubmed = 'cora_full ms_academic_cs ms_academic_phy'.split()

def model_selection(model, dataset):
    if model == 'gat' and dataset in datasets_similar_to_pubmed:
        dataset = 'pubmed' # take pubmed hyperparams and apply them to cora_full
    # if dataset == 'citeseer':
    #     dataset = 'cora' # take cora hyperparams and apply them to citeseer
    filename = f'reports/results/eval/{model}_val_{dataset}_undirected.csv'
    if not os.path.exists(filename):
        filename = f'reports/results/eval/{model}.csv'

    print(f'reading hyperparams from file {filename}')
    df = pd.read_csv(filename)
    df = df[df.arch == 'M']
    df = df[df.val_avg == df.val_avg.max()].reset_index().loc[0]
    df['dataset'] = dataset
    df['conv'] = model 
    return df

def extract_hyperparams(df_hyper, dataset, model):
    if model == 'gat' and dataset in datasets_similar_to_pubmed:
        dataset = 'pubmed' # take pubmed hyperparams and apply them to cora_full
    print(df_hyper)
    df_hyper = df_hyper[(df_hyper.dataset == dataset) & (df_hyper.conv == model)].reset_index().loc[0]
    attdrop = 0.3
    if 'attention_dropout' in df_hyper and df_hyper.attention_dropout is not None and not np.isnan(df_hyper.attention_dropout):
        attdrop = df_hyper.attention_dropout
    size = int(df_hyper.ch)
    if dataset == 'cora_full' and model == 'rgcn':
        size = 1
    return size, df_hyper.dropout, df_hyper.lr, df_hyper.wd, int(df_hyper.heads), attdrop

if __name__ == '__main__':
    args = parse_args()
    print(f'datasets: {args.datasets}')
    print(f'models: {args.models}')
    df_hyper = pd.DataFrame()
    for model in args.models:
        for dataset in args.datasets:
            df_hyper = df_hyper.append(model_selection(model,dataset))
    print(df_hyper)
    hubs_experiment = '_label_sbm' if args.label_sbm else ('_injected_edges' if args.injected_edges else '_injected_edges_sbm')
    if args.hubs_experiment is not None:
        hubs_experiment += '_' + args.hubs_experiment 
    for dataset in args.datasets:
        for directionality in args.directionalities:
            for model in args.models:
                print(f'{dataset}, {model}, {directionality}')
                val_out = f'reports/results/test_acc/{model}_{dataset}'\
                          f'{"_conf" if args.conf else ""}'\
                          f'{"_sbm" if args.sbm else ""}'\
                          f'{"_sbm_swap" if args.sbm_swap else ""}'\
                          f'{"_random" if args.random else ""}'\
                          f'{"_erdos" if args.erdos else ""}'\
                          f'{hubs_experiment if args.label_sbm else ""}'\
                          f'{hubs_experiment if args.injected_edges else ""}'\
                          f'{"_degree_cat" if args.injected_edges_degree_cat else ""}'\
                          f'{"_constant_nodes" if args.injected_edges_constant_nodes else ""}'\
                          f'{"_attack_target" if args.injected_edges_attack_target else ""}'\
                          f'{hubs_experiment if args.injected_edges_sbm else ""}'\
                          f'{"_flipped" if args.flipped else ""}'\
                          f'{"_removed_hubs" if args.removed_hubs else ""}'\
                          f'{"_added_2hop_edges" if args.added_2hop_edges else ""}'\
                          f'{("_" + directionality) if (directionality!="undirected") else ""}.csv'
                print(f'Evaluation will be saved to {val_out}')
                if os.path.exists(val_out):
                    df_val = pd.read_csv(val_out)
                    print(f'File {val_out} already exists, appeding to it')
                else:
                    print(f'Creating file {val_out}')
                    df_val = pd.DataFrame(
                        columns='conv arch ch dropout lr wd heads attention_dropout splits inits val_accs val_avg val_std'
                                ' test_accs test_avg test_std stopped elapsed'.split())
                size, dropout, lr, wd, heads, attention_dropout = extract_hyperparams(df_hyper,dataset,model)
                print(size)
                if args.sbm:
                    df_cur = eval_sbm(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                elif args.sbm_swap:
                    df_cur = eval_sbm_swap(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                elif args.conf:
                    df_cur = eval_conf(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                elif args.random:
                    df_cur = eval_random(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                elif args.erdos:
                    df_cur = eval_erdos(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                elif args.injected_edges:
                    df_cur = eval_injected_edges(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 5, range(1000,5001,1000), args.hubs_experiment)
                elif  args.injected_edges_degree_cat:
                    df_cur = eval_injected_edges_degree_cat(model, dataset, directionality, size, dropout, lr, wd,
                            heads, attention_dropout,
                            args.splits, args.runs, args.train_examples, args.val_examples, 5, 1000, 5)
                elif  args.injected_edges_constant_nodes:
                    df_cur = eval_injected_edges_constant_nodes(model, dataset, directionality, size, dropout, lr, wd,
                            heads, attention_dropout,
                            args.splits, args.runs, args.train_examples, args.val_examples, 5, 0.01, [4,8], 5)
                elif  args.injected_edges_attack_target:
                    df_cur = eval_injected_edges_attack_target(model, dataset, directionality, size, dropout, lr, wd,
                            heads, attention_dropout,
                            args.splits, args.runs, args.train_examples, args.val_examples, 5, 0.01, [4], 10)
                elif args.injected_edges_sbm:
                    df_cur = eval_injected_edges_sbm(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, 5, range(100,2001,100), args.hubs_experiment)
                elif args.label_sbm:
                    df_cur = eval_label_sbm(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples, args.hubs_experiment)
                elif args.flipped:
                    df_cur = eval_flipped(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples)
                elif args.removed_hubs:
                    df_cur = eval_removed_hubs(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples)
                elif args.added_2hop_edges:
                    df_cur = eval_added_2hop_edges(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples)
                else:
                    df_cur = eval_original(model, dataset, directionality, size, dropout, lr, wd, heads, attention_dropout, args.splits, args.runs,
                             args.train_examples, args.val_examples)
                df_val = pd.concat([df_val,df_cur])
                df_val.to_csv(val_out, index=False)
