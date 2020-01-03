

import torch
import torch.nn.functional as F
from models import MonoModel,BiModel,TriModel,TriPreModel,TriLateModel
import time

def run_and_eval_model(dataset,channels1,modelType,epochs=200):
    # training process (without batches/transforms)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MonoModel(modelType,dataset,channels1).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
    model.train() # to enter training phase
    for epoch in range(epochs):
        optimizer.zero_grad() # saw this a lot in the beginning, maybe resetting gradients (not to accumulate)
        out = model(data) # this calls the forward method apparently
        loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask]) # nice indexing, easy and short
        loss.backward() # magic: real back propagation step, takes care of the gradients and stuff
        optimizer.step() # maybe updates the params to be optimized
    model.eval() # enter eval phase
    _,pred = model(data).max(dim=1) # take prediction out of softmax
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    return acc

def eval_multiple(dataset,channels1,modelType,runs=100,epochs=50):
    start = time.time()
    accs = []
    for i in range(runs):
        accs.append(run_and_eval_model(dataset,channels1,modelType,epochs))
    elapsed_time = time.time() - start
    print('Elaplsed {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    return sum(accs)/len(accs),accs



def run_and_eval_bimodel(dataset,channels_st,channels_ts,modelType,epochs=200):
    # training process (without batches/transforms)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiModel(modelType,dataset,channels_st,channels_ts).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
    model.train() # to enter training phase
    for epoch in range(epochs):
        optimizer.zero_grad() # saw this a lot in the beginning, maybe resetting gradients (not to accumulate)
        out = model(data) # this calls the forward method apparently
        loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask]) # nice indexing, easy and short
        loss.backward() # magic: real back propagation step, takes care of the gradients and stuff
        optimizer.step() # maybe updates the params to be optimized
    model.eval() # enter eval phase
    _,pred = model(data).max(dim=1) # take prediction out of softmax
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    return acc
def eval_multiple_bi(dataset,channels_st,channels_ts,modelType,runs=100,epochs=50):
    start = time.time()
    accs = []
    for i in range(runs):
        accs.append(run_and_eval_bimodel(dataset,channels_st,channels_ts,modelType,epochs))
    elapsed_time = time.time() - start
    print('Elaplsed {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    return sum(accs)/len(accs),accs



def run_and_eval_trimodel(dataset,channels,channels_st,channels_ts,modelType,epochs=200):
    # training process (without batches/transforms)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TriModel(modelType,dataset,channels,channels_st,channels_ts).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
    model.train() # to enter training phase
    for epoch in range(epochs):
        optimizer.zero_grad() # saw this a lot in the beginning, maybe resetting gradients (not to accumulate)
        out = model(data) # this calls the forward method apparently
        loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask]) # nice indexing, easy and short
        loss.backward() # magic: real back propagation step, takes care of the gradients and stuff
        optimizer.step() # maybe updates the params to be optimized
    model.eval() # enter eval phase
    _,pred = model(data).max(dim=1) # take prediction out of softmax
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    return acc
def eval_multiple_tri(dataset,channels,channels_st,channels_ts,modelType,runs=100,epochs=50):
    start = time.time()
    accs = []
    for i in range(runs):
        accs.append(run_and_eval_trimodel(dataset,channels,channels_st,channels_ts,modelType,epochs))
    elapsed_time = time.time() - start
    print('Elaplsed {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    return sum(accs)/len(accs),accs

def run_and_eval_tripremodel(dataset,channels,channels_st,channels_ts,inter,modelType,epochs=200):
    # training process (without batches/transforms)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TriLateModel(modelType,dataset,channels,channels_st,channels_ts,inter).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
    model.train() # to enter training phase
    for epoch in range(epochs):
        optimizer.zero_grad() # saw this a lot in the beginning, maybe resetting gradients (not to accumulate)
        out = model(data) # this calls the forward method apparently
        loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask]) # nice indexing, easy and short
        loss.backward() # magic: real back propagation step, takes care of the gradients and stuff
        optimizer.step() # maybe updates the params to be optimized
    model.eval() # enter eval phase
    _,pred = model(data).max(dim=1) # take prediction out of softmax
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    return acc
def eval_multiple_tripre(dataset,channels,channels_st,channels_ts,inter,modelType,runs=100,epochs=50):
    start = time.time()
    accs = []
    for i in range(runs):
        accs.append(run_and_eval_tripremodel(dataset,channels,channels_st,channels_ts,inter,modelType,epochs))
    elapsed_time = time.time() - start
    print('Elaplsed {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    return sum(accs)/len(accs),accs