import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
from random import sample, seed
# from torch_geometric_node2vec import Node2Vec
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
from citation_loader import read_network
from shutil import copyfile
from network_split import NetworkSplitShcur
from sklearn.linear_model import LogisticRegression, LinearRegression


def load_embedding(embFile,featFile = None):
    xDict = {}
    yDict = {}
    
    rename_class = {}
    cnt_class = 0
    with open(featFile, 'r') as f:
        for line in f:
            s = line.split()
            id = s[0]
            emb = [float(x) for x in s[1:-1]]
            xDict[id] = emb
            if s[-1] not in rename_class:
                rename_class[s[-1]] = cnt_class
                cnt_class += 1
            yDict[id] = rename_class[s[-1]]

    rename_id={}
    cnt = 0
    for k in xDict.keys():
        rename_id[cnt] = k
        cnt += 1
            
    # Read embedding
    err = 0
    with open(embFile, 'r') as f:
        skip = True
        for line in f:
            s = line.split()
            if skip and len(s)==2:
                skip = False
                continue
            skip = False
            id = s[0]
            emb = [float(x) for x in s[1:]]
            if id in xDict:
                xDict[id] = xDict[id] + emb
            else:
                err +=1

    print(err)

    
    x = []
    target = []
    err = 0
    for i in range(cnt):
#         if (len(xDict[rename_id[i]])!=1561):
#             err+=1
#             continue
        x.append(np.array(xDict[rename_id[i]]))
        target.append(yDict[rename_id[i]])
    print(err)

    
    
    x = torch.tensor(np.array(x),dtype=torch.float)
    y = torch.tensor(np.array(target),dtype=torch.long)
    
    num_classes = len(set(target))
    df = pd.DataFrame(target)
    df.columns = ['target']
    train = []
    
    n = len(target)
    
    seed(1)
    for i in range(num_classes):
        train = train + sample(df[df['target']==i].index.values.tolist() ,20)
    rest = [i for i in range(n) if i not in train]
    val = rest[:500]
    test = rest[500:1500]
    train_ind = [1 if i in train else 0 for i in range(n)]
    val_ind = [1 if i in val else 0 for i in range(n)]
    test_ind = [1 if i in test else 0 for i in range(n)]

    data = Data(x=x, y=y)
    data.test_mask = torch.tensor(test_ind,dtype=torch.uint8)
    data.train_mask = torch.tensor(train_ind,dtype=torch.uint8)
    data.val_mask = torch.tensor(val_ind,dtype=torch.uint8)

    
    return data
    




class EmbeddingData(InMemoryDataset):
    
    embUrl = '../data/repeated-embedding'
    featsUrl = '../data/raw'

    def __init__(self, root, name, method, directed,reverse=False,initialization=None,nerd=None): #, transform=None, pre_transform=None):
        self.name = name
        self.method = method
        self.directed = directed
        self.reverse = reverse
        self.nerd = nerd
        
        if initialization is not None:
            self.embUrl = self.embUrl + "/" + str(initialization)
        
        super(EmbeddingData, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.{}.{}.emb'.format(self.name,self.method,('reversed' if self.reverse else 
                                      ('directed' if self.directed else 'undirected'))+('' if self.nerd is None else f'.{self.nerd}')),
                '{}.content'.format(self.name)]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        copyfile('{}/{}'.format(self.embUrl, self.raw_file_names[0]),'{}/{}'.format(self.raw_dir, self.raw_file_names[0]))
        copyfile('{}/{}/{}'.format(self.featsUrl, self.name, self.raw_file_names[1]),'{}/{}'.format(self.raw_dir, self.raw_file_names[1]))

    def process(self):
        emb_path = osp.join(self.raw_dir,self.raw_file_names[0])
        features_path = osp.join(self.raw_dir,self.raw_file_names[1])
        data = load_embedding(emb_path,features_path)
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def __repr__(self):
        return '{}()'.format(self.name)
    
    


def test(train_z, train_y, test_z, test_y, solver='lbfgs',
         multi_class='auto', *args, **kwargs):
    r"""Evaluates latent space quality via a logistic(?) regression downstream
    task."""
    clf = LogisticRegression(solver=solver, multi_class=multi_class,n_jobs=8, *args,
                           **kwargs).fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
#     clf = LinearRegression().fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
    return clf.score(test_z.detach().cpu().numpy(),
                     test_y.detach().cpu().numpy())

def eval_n2v(data,num_splits=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = data.x
    z = z.to(device)
    vals = []
    for i in range(num_splits):
        split = NetworkSplitShcur(data,early_examples_per_class=0,split_seed=i)
        val = test(z[split.train_mask], data.y[split.train_mask],
                         z[split.val_mask], data.y[split.val_mask], max_iter=100)
        vals.append(val)
    return vals

def test_n2v(data,num_splits=100,speedup=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = data.x
    z = z.to(device)
    tests = []
    for i in range(num_splits):
        split = NetworkSplitShcur(data,early_examples_per_class=0,split_seed=i,speedup=speedup)
#         print(f'split {i}')
        ts = test(z[split.train_mask], data.y[split.train_mask],
                         z[split.test_mask], data.y[split.test_mask], max_iter=150)
        tests.append(ts)
    return tests