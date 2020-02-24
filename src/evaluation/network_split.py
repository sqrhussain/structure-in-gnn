
import torch
import numpy as np
import random
import json
import os 

class NetworkSplitShchur:
    
    def __init__(self, data, train_examples_per_class=20,early_examples_per_class=20,
                 val_examples_per_class=30 , split_seed = 0):
        filename_prefix = data.name
        data = data[0]
        filename = f'../data/split_masks/{filename_prefix}-split{split_seed}.json'
        if os.path.exists(filename):
            with open(filename) as json_file:
                stored = json.load(json_file)
            self.train_mask = torch.BoolTensor(stored['train_mask'])
            self.early_mask = torch.BoolTensor(stored['early_mask'])
            self.val_mask = torch.BoolTensor(stored['val_mask'])
            self.test_mask = torch.BoolTensor(stored['test_mask'])
            return
        
        random.seed(split_seed)
        target = [x.item() for x in data.y]
        num_samples = len(target)
        indices = range(num_samples)
        rest = indices
        labels = set(target)
        num_classes = len(labels)

        train_idx = []
        for c in range(num_classes):
            train_idx += [x for x in random.sample([i for i in rest if target[i]==c],
                                                   train_examples_per_class)]
        self.train_mask = [1 if i in train_idx else 0 for i in range(num_samples)]

        rest = [x for x in rest if not self.train_mask[x]]
        early_idx = []
        if early_examples_per_class > 0:
            for c in range(num_classes):
                    early_idx += [x for x in random.sample([i for i in rest if target[i]==c],
                                                       early_examples_per_class)]
        self.early_mask = [1 if i in early_idx else 0 for i in range(num_samples)]

        rest = [x for x in rest if not self.early_mask[x]]
        val_idx = []
        for c in range(num_classes):
            val_idx += [x for x in random.sample([i for i in rest if target[i]==c],
                                                   val_examples_per_class)]

        self.val_mask = [1 if i in val_idx else 0 for i in range(num_samples)]

        rest = [x for x in rest if not self.val_mask[x]]
        self.test_mask = [1 if i in rest else 0 for i in indices]
        
        tostore = {'train_mask':self.train_mask,
                    'early_mask':self.early_mask,
                    'val_mask':self.val_mask,
                    'test_mask':self.test_mask}
        
        self.train_mask = torch.BoolTensor(self.train_mask)
        self.early_mask = torch.BoolTensor(self.early_mask)
        self.val_mask = torch.BoolTensor(self.val_mask)
        self.test_mask = torch.BoolTensor(self.test_mask)
        
        
        all_ones = self.train_mask | self.early_mask | self.val_mask | self.test_mask
        assert (all_ones.sum().item() == num_samples)

        
        with open(filename, 'w') as outfile:
            json.dump(tostore, outfile)
        
