import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import tqdm

from model import LanguageModel

from adasoft import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Benchmark for Adaptive Softmax')
parser.add_argument('--model', type=str, default='adasoft',
        help=('adasoft for Adaptive Softmax, '
            'linear for common linear projection'))

args = parser.parse_args()

with open('text8.train.pkl', 'rb') as f:
    data = pickle.load(f)

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)

    else:
        return tuple(repackage_hidden(v) for v in h)

def clip_global_norm(model, clip):
    norms = []
    total_norm = 0

    for p in model.parameters():
        norm = p.grad.data.norm()

        if norm > clip:
            p.grad.data.div_(max(norm, 1e-6) / clip)

input = data['input']
label = data['label']

vocab = len(data['worddic'])

if args.model == 'adasoft':
    adasoft = True

elif args.model == 'linear':
    adasoft = False

model = LanguageModel(vocab, 512, 512, 1,
        adaptive_softmax=adasoft, cutoff=[2000, 10000])
model.cuda()
optimizer = optim.Adagrad(model.parameters(), lr=0.1,
        lr_decay=1e-5, weight_decay=1e-5)

if adasoft:
    criterion = AdaptiveLoss([2000, 10000, vocab + 1])

else:
    criterion = nn.CrossEntropyLoss()

def train():
    pbar = tqdm.tqdm(zip(input, label))
    hidden = model.init_hidden(128)

    for X_batch, Y_batch in pbar:
        X_tensor = torch.from_numpy(X_batch).cuda()
        Y_tensor = torch.from_numpy(Y_batch.astype(np.int)).cuda()
        X_var, Y_var = Variable(X_tensor), Variable(Y_tensor.view(-1))
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(X_var, hidden, Y_var)
        loss = criterion(output, Y_var)
        loss.backward()
        clip_global_norm(model, 0.25)
        optimizer.step()
        pbar.set_description('Loss: {:.3f}'.format(loss.data[0]))

def test():
    pbar = tqdm.tqdm(zip(test_data['input'], test_data['label']))

    if adasoft:
        criterion = nn.NLLLoss(size_average=False)

    else:
        criterion = nn.CrossEntropyLoss(size_average=False)

    nllloss = 0
    hidden = model.init_hidden(128)

    for X_batch, Y_batch in pbar:
        X_tensor = torch.from_numpy(X_batch).cuda()
        Y_tensor = torch.from_numpy(Y_batch.astype(np.int)).cuda()
        X_var, Y_var = Variable(X_tensor), Variable(Y_tensor.view(-1))
        hidden = repackage_hidden(hidden)

        if adasoft:
            output, hidden = model.log_prob(X_var, hidden, Y_var)
            nllloss += criterion(Variable(output), Y_var).data[0]

        else:
            output, hidden = model(X_var, hidden, Y_var, training=False)
            nllloss += criterion(output, Y_var).data[0]


    loss = nllloss / (len(test_data['input']) * 128 * 20)

    print('Perplexity:', np.exp(loss))

    return loss

with open('text8.test.pkl', 'rb') as f:
    test_data = pickle.load(f)

for epoch in range(5):
    train()
    test()
