import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
import pandas as pd
from random import sample, seed
import numpy as np
from shutil import copyfile
import os.path as osp
import re
import json


class FilmTrust(InMemoryDataset):
    
    url = '../../rec-trust-net/netemb/data/filmtrust'

    def __init__(self, root, directed = True,reverse = False): #, transform=None, pre_transform=None):
        self.name = 'FilmTrust'
        self.directed = directed
        self.reverse = reverse
        super(FilmTrust, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['trust.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            copyfile('{}/{}'.format(self.url, name),'{}/{}'.format(self.raw_dir, name))

    def process(self):
        edge_path = osp.join(self.raw_dir,self.raw_file_names[0])
        data = read_torch_edgelist(edge_path)
        torch.save(self.collate([data]), self.processed_paths[0])
    
    def __repr__(self):
        return '{}()'.format(self.name)
    
class CiaoDVD(InMemoryDataset):
    
    url = '../../rec-trust-net/netemb/data/ciao'

    def __init__(self, root, directed = True,reverse = False): #, transform=None, pre_transform=None):
        self.name = 'CiaoDVD'
        self.directed = directed
        self.reverse = reverse
        super(CiaoDVD, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['trust_data.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            copyfile('{}/{}'.format(self.url, name),'{}/{}'.format(self.raw_dir, name))

    def process(self):
        edge_path = osp.join(self.raw_dir,self.raw_file_names[0])
        data = read_torch_edgelist(edge_path)
        torch.save(self.collate([data]), self.processed_paths[0])
    
    def __repr__(self):
        return '{}()'.format(self.name)
    
    
    
class Epinions(InMemoryDataset):
    
    url = '../../rec-trust-net/netemb/data/epinions'

    def __init__(self, root, directed = True,reverse = False): #, transform=None, pre_transform=None):
        self.name = 'Epinions'
        self.directed = directed
        self.reverse = reverse
        super(Epinions, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['trust_data.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            copyfile('{}/{}'.format(self.url, name),'{}/{}'.format(self.raw_dir, name))

    def process(self):
        edge_path = osp.join(self.raw_dir,self.raw_file_names[0])
        data = read_torch_edgelist(edge_path)
        torch.save(self.collate([data]), self.processed_paths[0])
    
    def __repr__(self):
        return '{}()'.format(self.name)