
import torch
import numpy as np
import random

class NetworkSplitShcur:
    
    def __init__(self, data, train_examples_per_class=20,early_examples_per_class=20,
                 val_examples_per_class=30 , split_seed = 0, speedup = False): # used speedup=False for each of Cora, Citeseer and Pubmed
        self.data = data
        
        random.seed(split_seed)
        target = [x.item() for x in self.data.y]
        num_samples = len(target)
        indices = range(num_samples)
        labels = set(target)
        num_classes = len(labels)
        
        train_idx = []
        for c in range(num_classes):
            train_idx += [x for x in random.sample([i for i in indices if target[i]==c],
                                                   train_examples_per_class)]
        self.train_mask = [1 if i in train_idx else 0 for i in range(num_samples)]

        rest = [x for x in indices if not self.train_mask[x]]
        early_idx = []
        if early_examples_per_class > 0:
            for c in range(num_classes):
                if speedup:
                    early_idx += [x for x in random.sample([i for i in rest if target[i]==c],
                                                       early_examples_per_class)]
                else:
                    early_idx += [x for x in random.sample([i for i in indices if i in rest and target[i]==c],
                                                       early_examples_per_class)]
        self.early_mask = [1 if i in early_idx else 0 for i in range(num_samples)]

        rest = [x for x in rest if not self.early_mask[x]]
        val_idx = []
        for c in range(num_classes):
            if speedup:
                val_idx += [x for x in random.sample([i for i in rest if target[i]==c],
                                                   val_examples_per_class)]
            else:
                val_idx += [x for x in random.sample([i for i in indices if i in rest and target[i]==c],
                                                   val_examples_per_class)]
        self.val_mask = [1 if i in val_idx else 0 for i in range(num_samples)]

        rest = [x for x in rest if not self.val_mask[x]]
        self.test_mask = [1 if i in rest else 0 for i in indices]
        
        
        self.train_mask = torch.BoolTensor(self.train_mask)
        self.early_mask = torch.BoolTensor(self.early_mask)
        self.val_mask = torch.BoolTensor(self.val_mask)
        self.test_mask = torch.BoolTensor(self.test_mask)
        
        
        all_ones = self.train_mask | self.early_mask | self.val_mask | self.test_mask
        assert (all_ones.sum().item() == num_samples)
        
        