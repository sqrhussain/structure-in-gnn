import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
from citation_loader import read_network
from shutil import copyfile

class CitationNetwork(InMemoryDataset):
    
    url = '../data/raw'

    def __init__(self, root, name, directed = True,reverse = False): #, transform=None, pre_transform=None):
        self.name = name
        self.directed = directed
        self.reverse = reverse
        super(CitationNetwork, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.cites'.format(self.name),'{}.content'.format(self.name)]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
#         pass
        for name in self.raw_file_names:
            copyfile('{}/{}/{}'.format(self.url, self.name, name),'{}/{}'.format(self.raw_dir, name))

    def process(self):
        edge_path = osp.join(self.raw_dir,self.raw_file_names[0])
        features_path = osp.join(self.raw_dir,self.raw_file_names[1])
        data = read_network(features_path,edge_path,self.directed,self.reverse)
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def sample_mask(self,index, num_nodes, inv=False):
        if inv:
            mask = torch.ones((num_nodes, ), dtype=torch.uint8)
            mask[index] = 0

        mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
        mask[index] = 1
        return mask
    
    def __repr__(self):
        return '{}()'.format(self.name)

class ConfigurationModelCitationNetwork(CitationNetwork):
    
    conf_model_url = '../data/processed/conf_model'
    def __init__(self, root, name, directed = True,reverse = False): #, transform=None, pre_transform=None):
        super(ConfigurationModelCitationNetwork, self).__init__(root, name, directed,reverse)
        self.directed = directed
        self.reverse = reverse

    def download(self):
#         pass
        copyfile('{}/{}/{}'.format(self.url, self.name, self.raw_file_names[1]),'{}/{}'
                 .format(self.raw_dir, self.raw_file_names[1]))
        copyfile('{}/{}/{}'.format(self.conf_model_url, self.name, self.raw_file_names[0]),'{}/{}'
                 .format(self.raw_dir, self.raw_file_names[0]))
            
            
class CocitationNetwork(InMemoryDataset):
    
    feats_dir = '../data/raw'
    net_dir = '../data/processed'
    def __init__(self, root, name, net_type): #, transform=None, pre_transform=None):
        self.name = name
        self.net_type = net_type
        super(CocitationNetwork, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.{}'.format(self.name,self.net_type),'{}.content'.format(self.name)]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        copyfile('{}/{}/{}.{}'.format(self.net_dir, self.name, self.name,self.net_type),
                 '{}/{}.{}'.format(self.raw_dir, self.name,self.net_type))
        copyfile('{}/{}/{}.content'.format(self.feats_dir, self.name, self.name),
                 '{}/{}.content'.format(self.raw_dir, self.name))

    def process(self):
        edge_path = osp.join(self.raw_dir,self.raw_file_names[0])
        features_path = osp.join(self.raw_dir,self.raw_file_names[1])
        data = read_network(features_path,edge_path,self.directed ,self.reverse)
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def sample_mask(self,index, num_nodes, inv=False):
        if inv:
            mask = torch.ones((num_nodes, ), dtype=torch.uint8)
            mask[index] = 0

        mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
        mask[index] = 1
        return mask
    
    def __repr__(self):
        return '{}()'.format(self.name)
    
