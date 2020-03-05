import torch
from torch_geometric.data import InMemoryDataset
import os.path as osp
from src.data.data_load_helper import read_network, load_embedding
from shutil import copyfile


class GraphDataset(InMemoryDataset):


    def __init__(self, root, name, edge_path, attributes_path, directed = True,reverse = False):
        self.name = name
        
        self.edge_path = edge_path
        self.attributes_path = attributes_path

        self.directed = directed
        self.reverse = reverse
        
        super(GraphDataset, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        return

    def process(self):
        data = read_network(self.attributes_path,self.edge_path,self.directed,self.reverse)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class FeatureOnlyData(InMemoryDataset):
    
    def __init__(self, root, name, attributes_path):        
        
        self.name = name
        self.attributes_path = attributes_path

        super(FeatureOnlyData, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data = load_embedding(embFile = None,featFile = self.attributes_path)
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def __repr__(self):
        return '{}()'.format(self.name)

    @property
    def processed_file_names(self):
        return 'data.pt'

    """ No need to copy raw files """
    @property
    def raw_file_names(self):
        return []

    """ No need to copy raw files """
    def download(self):
        return



class EmbeddingData(InMemoryDataset):
    
    def __init__(self, root, name, embedding_path, attributes_path):

        self.name = name
        self.embedding_path = embedding_path
        self.attributes_path = attributes_path
                
        super(EmbeddingData, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        data = load_embedding(emb_path,features_path)
        torch.save(self.collate([data]), self.processed_paths[0])
        
    def __repr__(self):
        return '{}()'.format(self.name)

    @property
    def processed_file_names(self):
        return 'data.pt'

    """ No need to copy raw files """
    @property
    def raw_file_names(self):
        return []

    """ No need to copy raw files """
    def download(self):
        return


class CitationNetwork(InMemoryDataset):
    
    url = '../data/raw'

    def __init__(self, root, name, directed = True,reverse = False):
        print("WARNING: this class is obsolete. Please use GraphDataset instead. thancc")
        self.name = name
        self.directed = directed
        self.reverse = reverse
        
        if self.reverse:
            root = root+'-reversed'
        elif self.directed:
            root = root+'-directed'

        super(CitationNetwork, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.cites'.format(self.name),'{}.content'.format(self.name)]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
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
