import matplotlib.pyplot as plt
import torch
from cora_loader import CitationNetwork,CocitationNetwork,ConfigurationModelCitationNetwork
from multi_layered_model import MonoModel,BiModel,TriModel
from torch_geometric.nn import GCNConv,SAGEConv,GATConv
import time
import torch.nn.functional as F
import copy
import numpy as np
import random
import warnings
import pandas as pd
from network_split import NetworkSplitShcur



def run_and_eval_model(dataset,channels,modelType,architecture,
                       lr,wd,heads,dropout,
                       epochs=200,
                       split_seed=0,init_seed=0):
    # training process (without batches/transforms)
    
    # Uncomment to test sources of randomness
    # we assume that the only sources of randomness are the data split and the initialization.
    # We can probably use this line when we need the results to be reproducible
    torch.manual_seed(init_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)

        
    if modelType == GATConv:
        model = architecture(dataset,channels,dropout=dropout,heads=heads).to(device)
    else:
        model = architecture(modelType,dataset,channels,dropout).to(device)
    
    
    print('split started')
    split = NetworkSplitShcur(data,early_examples_per_class=0,split_seed=split_seed,speedup=(dataset.name == 'cora_full'))
    print('split finished')
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    model.train() # to enter training phase
    maxacc=-1
    chosen = None
    accs = []
    stopped_at = epochs
    for epoch in range(epochs):
        optimizer.zero_grad() # saw this a lot in the beginning, maybe resetting gradients (not to accumulate)
        out = model(data) # this calls the forward method apparently
        loss = F.nll_loss(out[split.train_mask],data.y[split.train_mask]) # nice indexing, easy and short
        loss.backward() # magic: real back propagation step, takes care of the gradients and stuff
        optimizer.step() # maybe updates the params to be optimized
        
        model.eval()
        _,pred = model(data).max(dim=1) # take prediction out of softmax
        correct = float(pred[split.val_mask].eq(data.y[split.val_mask]).sum().item())
        acc = correct / float(split.val_mask.sum().item())
        accs.append(acc)
        if acc > maxacc:
            maxacc = acc
            chosen=copy.deepcopy(model)
        if epoch > 10 and acc*10 < sum(accs[-11:-1]):
#             print('stopped at epoch {}'.format(epoch))
            stopped_at = epoch
            break
        model.train()
    chosen.eval() # enter eval phase
    _,pred = chosen(data).max(dim=1) # take prediction out of softmax
    correct = float(pred[split.val_mask].eq(data.y[split.val_mask]).sum().item())
    val_acc = correct / float(split.val_mask.sum().item())
    
    correct = float(pred[split.test_mask].eq(data.y[split.test_mask]).sum().item())
    test_acc = correct / float(split.test_mask.sum().item())
    
    return val_acc,test_acc,stopped_at

def eval_multiple(dataset,channels,modelType,architecture,
                  lr,wd,heads, dropout,
                  runs=100,epochs=50,split_seed = 0):
    start = time.time()
    val_accs = []
    test_accs = []
    stoppeds = []
    for i in range(runs):
        val_acc,test_acc, stopped = run_and_eval_model(dataset,channels,modelType,architecture,lr=lr,wd=wd,epochs=epochs,
                                       heads=heads, dropout=dropout,
                                       split_seed=split_seed, init_seed=i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        stoppeds.append(stopped)
    elapsed_time = time.time() - start

    return val_accs,test_accs,stoppeds


def eval_archs(dataset,
               conv,
               channel_size,dropout,lr,wd,heads,
               models=[MonoModel, BiModel, TriModel],
              num_splits=100,num_runs=20,df_val=None):
#     if val_out == None:
#         val_out = f'val_{conv.__name__}.csv'
    
    
    chs = 0
    for model in models:
        chs+=1
        val_accs = []
        test_accs = []
        stoppeds = []
        fileout = f'acc_{num_splits}_{num_runs}/{conv.__name__}_{model.__name__}_ch{channel_size}_lr{lr}_wd{wd}_dropout{dropout}_heads{heads}.out'
        print(f'>> {fileout}')
        with open(fileout,'w') as f:
            start = time.time()
            for seed in range(num_splits):
                val_acc,test_acc,stopped = eval_multiple(dataset,[channel_size//chs//heads],conv,runs=num_runs,epochs=200,split_seed=seed,
                                                            architecture=model,lr=lr,wd=wd,heads=heads,dropout=dropout)
#                 f.write('{} {} accuracy: {:.2f} +-{:.1f}\n'.format(conv.__name__,model.__name__,monogcn*100,monogcnstd*100))
                val_accs += val_acc
                test_accs += test_acc
                stoppeds+= stopped
                
            val_avg = np.array(val_accs).mean()
            val_std = np.array(val_accs).std()
            
            test_avg = np.array(test_accs).mean()
            test_std = np.array(test_accs).std()
            f.write('{} {} accuracy: {:.2f} +-{:.1f}\n'.format(conv.__name__,model.__name__,val_avg*100,val_std*100))
        elapsed = time.time() - start
        df_val = df_val.append({'conv':conv.__name__,'arch':model.__name__[0],'ch':channel_size,
                               'dropout':dropout,'lr':lr,'wd':wd,'heads':heads,
                                'splits':num_splits,'inits':num_runs,
                                'val_accs':val_accs,'val_avg':val_avg,'val_std':val_std,
                                'test_accs':test_accs,'test_avg':test_avg,'test_std':test_std,
                                'stopped':stoppeds,'elapsed':elapsed},ignore_index=True)
#         df_val.to_csv(val_out,index=False)
#         print(df_val.shape)
    return df_val
    
    
