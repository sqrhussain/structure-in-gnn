import torch
from src.evaluation.network_split import NetworkSplitShchur
from sklearn.linear_model import LogisticRegression
from src.data.data_loader import EmbeddingData


def test_embedding(train_z, train_y, test_z, test_y, solver='lbfgs',
                   multi_class='ovr', seed=None, *args, **kwargs):
    r"""Evaluates latent space quality via a logistic(?) regression downstream
    task."""
    z = train_z.detach().cpu().numpy()
    y = train_y.detach().cpu().numpy()
    logreg = LogisticRegression(solver=solver, multi_class=multi_class, random_state=seed, n_jobs=1, *args, **kwargs)
    clf = logreg.fit(z, y)
    return clf.score(test_z.detach().cpu().numpy(),
                     test_y.detach().cpu().numpy())


def eval_method(data, num_splits=100,
                train_examples = 20, val_examples = 30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = data[0].x
    z = z.to(device)
    vals = []
    for i in range(num_splits):
        split = NetworkSplitShchur(data, train_examples_per_class=train_examples,early_examples_per_class=0,
                 val_examples_per_class=val_examples, split_seed=i)
        val = test_embedding(z[split.train_mask], data[0].y[split.train_mask],
                             z[split.val_mask], data[0].y[split.val_mask], max_iter=100)
        vals.append(val)
    return vals


def test_method(data, num_splits=100,
                train_examples = 20, val_examples = 30,seed=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = data[0].x
    z = z.to(device)
    tests = []
    for i in range(num_splits):
        split = NetworkSplitShchur(data, train_examples_per_class=train_examples,early_examples_per_class=0,
                 val_examples_per_class=val_examples, split_seed=i)
        ts = test_embedding(z[split.train_mask], data[0].y[split.train_mask],
                            z[split.test_mask], data[0].y[split.test_mask], max_iter=100,seed=seed)
        tests.append(ts)
    return tests


def report_test_acc_unsupervised_embedding(tmp,dataset,embfile,attrfile,
                        num_splits, train_examples, val_examples):
    tests = []
    emb = EmbeddingData(tmp, dataset,embfile,attrfile)
    print(f'started test {dataset}')
    test = test_method(emb, num_splits=num_splits, train_examples = train_examples, val_examples = train_examples)
    print(str(test) + '\n')
    tests = tests + test
    return tests