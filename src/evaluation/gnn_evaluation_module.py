import matplotlib.pyplot as plt
import torch
from src.models.multi_layered_model import MonoModel, BiModel, TriModel
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP
import time
import torch.nn.functional as F
import copy
import numpy as np
import random
import warnings
import pandas as pd
from src.evaluation.network_split import NetworkSplitShchur
from sklearn.metrics import f1_score


def build_complete_graph_per_class(data,split):
    target = [x.item() for x in data.y]
    labels = set(target)
    num_nodes = len(target)
    all_rows = []
    all_cols = []
    for c in labels:
        labeled_nodes = [u for u in range(num_nodes) if split.train_mask[u] and target[u] == c]
        edges = [[u, v] for u in labeled_nodes for v in labeled_nodes if u > v]
        rows = [e[0] for e in edges]
        cols = [e[1] for e in edges]
        all_rows = all_rows + rows
        all_cols = all_cols + cols

    return [all_rows,all_cols]

def train_gnn(dataset, channels, modelType, architecture,
              lr, wd, heads, dropout, attention_dropout,
              epochs,
              train_examples, val_examples,
              split_seed=0, init_seed=0,
              test_score=False, actual_predictions=False, add_complete_edges=False):
    # training process (without batches/transforms)

    # we assume that the only sources of randomness are the data split and the initialization.
    torch.manual_seed(init_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    split = NetworkSplitShchur(dataset, train_examples_per_class=train_examples,early_examples_per_class=0,
                 val_examples_per_class=val_examples, split_seed=split_seed)

    # for each class, add a complete graph of its labeled nodes
    if add_complete_edges:
        additional_edge_index = build_complete_graph_per_class(data, split)
        data.edge_index = torch.cat([data.edge_index, torch.tensor(additional_edge_index).to(device)], 1)

    if modelType == GATConv:
        model = architecture(dataset, channels, dropout=dropout, heads=heads,attention_dropout=attention_dropout).to(device)
    elif modelType == APPNP:
        model = architecture(dataset, channels, dropout=dropout).to(device)
    else:
        model = architecture(modelType, dataset, channels, dropout).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()  # to enter training phase
    maxacc = -1
    chosen = None
    accs = []
    # f1s = []
    stopped_at = epochs
    for epoch in range(epochs):
        optimizer.zero_grad()  # saw this a lot in the beginning, maybe resetting gradients (not to accumulate)
        out = model(data)  # this calls the forward method apparently
        loss = F.nll_loss(out[split.train_mask], data.y[split.train_mask])  # nice indexing, easy and short
        loss.backward()  # magic: real back propagation step, takes care of the gradients and stuff
        optimizer.step()  # maybe updates the params to be optimized

        model.eval()
        _, pred = model(data).max(dim=1)  # take prediction out of softmax
        correct = float(pred[split.val_mask].eq(data.y[split.val_mask]).sum().item())
        acc = correct / float(split.val_mask.sum().item())
        # f1 = f1_score(data.y[split.val_mask].detach().cpu().numpy(),pred[split.val_mask].detach().cpu().numpy(),average='micro')
        accs.append(acc)
        # f1s.append(f1)
        if acc > maxacc:
            maxacc = acc
            chosen = copy.copy(model)
        if epoch > 10 and acc * 10 < sum(accs[-11:-1]):
            stopped_at = epoch
            break
        model.train()
    chosen.eval()  # enter eval phase
    _, pred = chosen(data).max(dim=1)  # take prediction out of softmax
    correct = float(pred[split.val_mask].eq(data.y[split.val_mask]).sum().item())
    val_acc = correct / float(split.val_mask.sum().item())
    # val_f1 = f1_score(data.y[split.val_mask].detach().cpu().numpy(),pred[split.val_mask].detach().cpu().numpy(),average='micro')
    if test_score:
        correct = float(pred[split.test_mask].eq(data.y[split.test_mask]).sum().item())
        test_acc = correct / float(split.test_mask.sum().item())
        # test_f1 = f1_score(data.y[split.test_mask].detach().cpu().numpy(),pred[split.test_mask].detach().cpu().numpy(),average='micro')
        return val_acc, stopped_at, test_acc
    return val_acc, stopped_at, []


def train_gnn_multiple_runs(dataset, channels, modelType, architecture,
                            lr, wd, heads, dropout, attention_dropout,
                            train_examples, val_examples,
                            runs, epochs, split_seed=0,
                            test_score=False, actual_predictions=False):
    start = time.time()
    val_accs = []
    test_accs = []
    stoppeds = []
    for i in range(runs):
        val_acc, stopped, test_acc = train_gnn(dataset, channels, modelType, architecture, lr=lr, wd=wd, epochs=epochs,
                                               heads=heads, dropout=dropout, attention_dropout=attention_dropout,
                                               split_seed=split_seed, init_seed=i, test_score=test_score,
                                               actual_predictions=actual_predictions,
                                               train_examples = train_examples, val_examples = val_examples)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        stoppeds.append(stopped)
    elapsed_time = time.time() - start

    print(f'val: {val_accs}')
    print(f'tes: {test_accs}')
    print(f'sto: {stoppeds}')

    return val_accs, stoppeds, test_accs


def eval_gnn(dataset,
             conv,
             channel_size, dropout, lr, wd, heads,attention_dropout=0.3,
             models=[MonoModel, BiModel, TriModel],
             num_splits=100, num_runs=20,
             test_score=False, actual_predictions=False,
             train_examples = 20, val_examples = 30):
    columns = ['conv', 'arch', 'ch', 'dropout', 'lr', 'wd', 'heads', 'attention_dropout', 'splits', 'inits',
               'val_accs', 'val_avg', 'val_std', 'stopped', 'elapsed']
    if test_score:
        columns += ['test_accs', 'test_avg', 'test_std']
    df_val = pd.DataFrame(columns=columns)
    channels = 0
    for model in models:
        channels += 1
        val_accs = []
        test_accs = []
        stoppeds = []
        start = time.time()
        for seed in range(num_splits):
            val_acc, stopped, test_acc = train_gnn_multiple_runs(dataset, [channel_size // channels // heads], conv,
                                                                 runs=num_runs, epochs=200, split_seed=seed,
                                                                 architecture=model, lr=lr, wd=wd, heads=heads,
                                                                 attention_dropout=attention_dropout,
                                                                 dropout=dropout,test_score=test_score,
                                                                 train_examples = train_examples, val_examples = val_examples)
            val_accs += val_acc
            test_accs += test_acc
            stoppeds += stopped

        val_avg = np.array(val_accs).mean()
        val_std = np.array(val_accs).std()

        elapsed = time.time() - start
        row = {'conv': conv.__name__, 'arch': model.__name__[0], 'ch': channel_size,
               'dropout': dropout, 'lr': lr, 'wd': wd, 'heads': heads, 'attention_dropout':attention_dropout,
               'splits': num_splits, 'inits': num_runs,
               'val_accs': val_accs, 'val_avg': val_avg, 'val_std': val_std,
               'stopped': stoppeds, 'elapsed': elapsed}
        if test_score:
            row['test_accs'] = test_accs
            row['test_avg'] = np.array(test_accs).mean()
            row['test_std'] = np.array(test_accs).std()
        df_val = df_val.append(row, ignore_index=True)
    return df_val


