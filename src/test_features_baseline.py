import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from src.evaluation.embedding_evaluation_module import test_n2v
from src.data.data_loader import FeatureOnlyData
import argparse

num_splits = 100

def report_test_acc_unsupervised_embedding(dataset,speedup=False):
    tests = []
    emb = FeatureOnlyData(f'/tmp/{dataset.capitalize()}FeatsOnly',dataset)
    print(f'started test {dataset}')
    test = test_n2v(emb[0],num_splits=num_splits,speedup=speedup)
    print(str(test) + '\n')
    tests = tests + test
    return tests


val_out = f'test_acc/features-only-baseline.csv'
if os.path.exists(val_out):
	test_acc = pd.read_csv(val_out)
else:
	test_acc = pd.DataFrame(columns='method dataset test_acc test_avg test_std'.split())

datasets ='cora citeseer PubMed cora_full'.split()
for dataset in datasets:
	tests = report_test_acc_unsupervised_embedding(dataset=dataset,speedup=(dataset=='cora_full'))
	test_acc = test_acc.append({'method': 'features-only-baseline','dataset':dataset,
	                'test_acc':tests, 'test_avg':np.mean(tests), 'test_std':np.std(tests)},ignore_index=True)
	test_acc.to_csv(val_out,index=False)
