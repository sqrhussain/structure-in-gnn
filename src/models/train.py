from src.apply_gnn_to_datasets import eval_original, eval_conf, eval_sbm
import argparse


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

    parser.add_argument('--dataset',
                        nargs='+',
                        help='datasets to process, e.g., --dataset \'cora pubmed\'')
    parser.add_argument('--models',
                        nargs='+',
                        help='models to evaluate, e.g., --models \'gcn sage gat\'')
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


if __name__ == '__main__':
    args = parse_args()
    # for model in args.models:
    #     for dataset in args.datasets:
    #         # find the best performing setting
    #         pass
    for directionality in ['directed', 'reversed', 'undirected']:
        for model in args.models:
            for dataset in args.datasets:
                if args.sbm:
                    eval_sbm(model, dataset, directionality, 96, 0.4, 0.005, 0.1, 1, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                elif args.conf:
                    eval_conf(model, dataset, directionality, 96, 0.4, 0.005, 0.1, 1, args.splits, args.runs,
                             args.train_examples, args.val_examples, 10)
                else:
                    eval_original(model, dataset, directionality, 96, 0.4, 0.005, 0.1, 1, args.splits, args.runs,
                             args.train_examples, args.val_examples)
