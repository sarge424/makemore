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
block_size = 3

for word in names[:5]:
    context = [0] * block_size
    for ch in word + '.':
        X.append(context)
        Y.append(stoi[ch])
        context = context[1:] + [stoi[ch]]
        
n_ex = len(Y)
X = torch.tensor(X)
Y = torch.tensor(Y)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

for _ in range(20):
    #forward pass
    emb = C[X]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)
    print(loss.item())

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -0.1 * p.grad
