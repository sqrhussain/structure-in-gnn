## PLEASE RUN FROM PROJECT ROOT
import scipy
import wget
from gnnbench.data.make_dataset import get_dataset

data_remote = 'https://github.com/shchur/gnn-benchmark/blob/master/data/npz/cora_full.npz'
data_local = 'data/raw/cora_full.npz'
wget.download(data_remote,data_local)

graph_adj, node_features, labels = get_dataset('cora_full',data_local,True,_log=None)

with open('data/processed/cora_format/cora_full/cora_full.cites','w') as f:
    cx = scipy.sparse.coo_matrix(graph_adj)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        f.write(f'{i} {j}\n')


feats = node_features.todense()
labs = labels.argmax(axis=1)
with open('data/processed/cora_format/cora_full/cora_full.content','w') as f:
    for u in range(feats.shape[0]):
        f.write(f'{u} ')
        f.write(' '.join([str(int(x)) for x in feats.A[u]]))
        f.write(f' {labs[u]}\n')