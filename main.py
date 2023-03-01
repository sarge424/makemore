import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

#read file and get names
names = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(names))))

#define int <-> str mappings
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}

#create training sets
X, Y = [], []
block_size = 5

for word in names[:3]:
    context = [0] * block_size
    print(word)
    for ch in word + '.':
        X.append(context)
        Y.append(stoi[ch])
        
        print(''.join(itos[x] for x in context), '->', ch)
        context = context[1:] + [stoi[ch]]
        
X = torch.tensor(X)
Y = torch.tensor(Y)

#build lookup table
C = torch.randn((27, 2))