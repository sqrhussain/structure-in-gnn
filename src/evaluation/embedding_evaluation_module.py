import torch
from src.evaluation.network_split import NetworkSplitShchur
from sklearn.linear_model import LogisticRegression


def test_embedding(train_z, train_y, test_z, test_y, solver='lbfgs',
                   multi_class='ovr'):
    r"""Evaluates latent space quality via a logistic(?) regression downstream
    task."""
    z = train_z.detach().cpu().numpy()
    y = train_y.detach().cpu().numpy()
    logreg = LogisticRegression(solver=solver, multi_class=multi_class, n_jobs=1)
    clf = logreg.fit(z, y)
    return clf.score(test_z.detach().cpu().numpy(),
                     test_y.detach().cpu().numpy())


def eval_method(data, num_splits=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = data[0].x
    z = z.to(device)
    vals = []
    for i in range(num_splits):
        split = NetworkSplitShchur(data, early_examples_per_class=0, split_seed=i)
        val = test_embedding(z[split.train_mask], data[0].y[split.train_mask],
                             z[split.val_mask], data[0].y[split.val_mask], max_iter=100)
        vals.append(val)
    return vals


def test_method(data, num_splits=100, speedup=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = data[0].x
    z = z.to(device)
    tests = []
    for i in range(num_splits):
        split = NetworkSplitShchur(data, name, early_examples_per_class=0, split_seed=i, speedup=speedup)
        ts = test_embedding(z[split.train_mask], data[0].y[split.train_mask],
                            z[split.test_mask], data[0].y[split.test_mask], max_iter=150)
        tests.append(ts)
    return tests
