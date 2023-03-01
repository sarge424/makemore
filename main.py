import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from engine import Bigram

# --------------------------------------------------
#INTIALIZATION
#--------------------------------------------------

#read file and get names
names = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(names))))


#define int <-> str mappings
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}

#generate xs and ys
xs = []
ys = []
for name in names:
    word = '.' + name + '.'
    for c1,c2 in zip(word, word[1:]):
        i1 = stoi[c1]
        i2 = stoi[c2]
        xs.append(i1)
        ys.append(i2)

#--------------------------------------------------
#BIGRAM MODEL IMPLEMENTATION
#--------------------------------------------------
'''
#create bigram model and train it
m = Bigram(len(chars) + 1, bias=1, seed=2147483647)
m.train(xs, ys)

#sample from the model
for _ in range(5):
    outs = []
    i = 0
    while True:
        i = m(i).item()
        outs.append(itos[i])
        if i == 0:
            break
    print(''.join(outs))
        
#found to be 2.454094171524048
print(f'log likelihood={m.log_likelihood(xs, ys)}')
'''     
#--------------------------------------------------
#NEURAL NETWORK IMPLEMENTATION
#--------------------------------------------------

#set up generator
g = torch.Generator().manual_seed(2147483647)

n = len(xs)
xs = torch.tensor(xs, dtype=torch.int64)
ys = torch.tensor(ys, dtype=torch.int64)

print('number of examples:', n)

#single linear layer of 27 neurons
W = torch.randn((27, 27), generator=g, requires_grad=True)

#gradient descent
for d in range(100):
    #forward pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    #calculate loss
    loss = -probs[torch.arange(n), ys].log().mean()
    if d % 10 == 0:
        print(d, '- loss:', loss.item())

    #backward pass
    W.grad = None #more efficient
    loss.backward()

    #update
    W.data += -50 * W.grad
    
#sample from the network
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    outs = []
    i = 0
    while True:
        xenc = F.one_hot(torch.tensor((i)), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum()
        
        i = torch.multinomial(probs, 1, replacement=True, generator=g).item()
        outs.append(itos[i])
        if i == 0:
            break
    print(''.join(outs))