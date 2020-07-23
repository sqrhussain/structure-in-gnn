import pandas as pd
import numpy as np
import os
from src.evaluation.embedding_evaluation_module import test_method
from src.data.data_loader import FeatureOnlyData

num_splits = 100


def report_test_acc_unsupervised_embedding(dataset,seed):
    tests = []
    emb = FeatureOnlyData(f'data/tmp/{dataset.capitalize()}FeatsOnly', dataset,
                          f'data/graphs/processed/{dataset}/{dataset}.content')
    print(f'started test {dataset}')
    test = test_method(emb, num_splits=num_splits, train_examples = train_count(dataset), val_examples = val_count(dataset), seed=seed)
    print(str(test) + '\n')
    tests = tests + test
    return tests

val_out = f'reports/results/test_acc/features-only-baseline.csv'

print(f'reading {val_out}')
if os.path.exists(val_out):
    test_acc = pd.read_csv(val_out)
else:
    test_acc = pd.DataFrame(columns='method dataset test_acc test_avg test_std'.split())
    
# datasets = 'cora citeseer pubmed twitter webkb cora_full'.split()
datasets = 'amazon_electronics_computers amazon_electronics_photo ms_academic_cs ms_academic_phy'.split()

def train_count(dataset):
    if 'webkb' in dataset:
        return 10
    return 20
def val_count(dataset):
    if 'webkb' in dataset:
        return 15
    return 30


for dataset in datasets:
    print(f'evaluating {dataset}')
    tests = []
    inits = 20
    for seed in range(inits):
        tests = tests + report_test_acc_unsupervised_embedding(dataset=dataset,seed=seed)
    test_acc = test_acc.append({'method': 'features-only-baseline', 'dataset': dataset,
                                'test_acc': tests, 'test_avg': np.mean(tests), 'test_std': np.std(tests)},
                               ignore_index=True)
    test_acc.to_csv(val_out, index=False)
