from numpy import block
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

#hyperparams
block_size = 3
dimensions = 10
hidden_layer_size = 200
minibatch_size = 32

#create training sets
def create_dataset(words, bs):
    X, Y = [], []

    for word in words:
        context = [0] * bs
        for ch in word + '.':
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

import random
random.seed(42)
random.shuffle(names)
n1 = int(0.8*len(names))
n2 = int(0.9*len(names))

Xtr, Ytr = create_dataset(names[:n2], block_size)
#Xdev, Ydev = create_dataset(names[n1:n2])
Xtest, Ytest = create_dataset(names[n2:], block_size)

#init parameters
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((27, dimensions),                              generator=g)
W1 = torch.randn((block_size * dimensions, hidden_layer_size), generator=g) * 0.1
#b1 = torch.randn(hidden_layer_size,                            generator=g) * 0.01
W2 = torch.randn((hidden_layer_size, 27),                      generator=g) * 0.01
b2 = torch.randn(27,                                           generator=g) * 0

bngain = torch.ones((1, hidden_layer_size))
bnbias = torch.zeros((1, hidden_layer_size))

bnmean_running = torch.zeros((1, hidden_layer_size))
bnstd_running = torch.ones((1, hidden_layer_size))

parameters = [C, W1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True


#gradient descent
ltr = []
iters = []

print('training...')
for i in range(200000):
    #minibatch
    bi = torch.randint(0, Xtr.shape[0], (minibatch_size,))
    
    #forward pass
    emb = C[Xtr[bi]]
    hpre = emb.view(-1, block_size * dimensions) @ W1
    #batch normalize
    bnmeani = hpre.mean(0, keepdim=True)
    bnstdi = hpre.std(0, keepdim=True)
    hbn = (hpre - bnmeani) / (bnstdi + 1e-5)    
    #continue
    h = torch.tanh(bngain * hbn + bnbias)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[bi])

    with torch.no_grad():
        bnmean_running = bnmean_running * 0.999 + bnmeani * 0.001
        bnstd_running = bnstd_running * 0.999 + bnstdi * 0.001

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    #descend
    lr = 0.1
    if i > 100000:
        lr = 0.01
        
    for p in parameters:
        p.data += -lr * p.grad
        
    #stats
    if i % 100 == 0:
        ltr.append(loss.log10().item())
        iters.append(i)
        
    if i % 5000 == 0:
        print('.', i, '->',loss.item())
        

#calculate final loss
emb = C[Xtr]
hpre = emb.view(-1, block_size * dimensions) @ W1
#batch normalize
hbn = (hpre - bnmean_running) / (bnstd_running + 1e-5)
#continue
h = torch.tanh(bngain * hbn + bnbias)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print('training set loss:', loss.item())

emb = C[Xtest]
hpre = emb.view(-1, block_size * dimensions) @ W1
#batch normalize
hbn = (hpre - bnmean_running) / (bnstd_running + 1e-5)
#continue
h = torch.tanh(bngain * hbn + bnbias)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytest)
print('test set loss:', loss.item())

#sample from the model
for _ in range(20):
    outs = []
    context = [0] * block_size
    i = 0
    while True:
        emb = C[torch.tensor(context)]
        hpre = emb.view(-1, block_size * dimensions) @ W1
        #batch normalize
        hbn = (hpre - bnmean_running) / (bnstd_running + 1e-5)
        #continue
        h = torch.tanh(bngain * hbn + bnbias)
        logits = h @ W2 + b2
        prob = F.softmax(logits, dim=1)
        i = torch.multinomial(prob, 1, generator=g).item()
        context = context[1:] + [i]
        outs.append(i)
        if i == 0:
            break
    print(''.join(itos[x] for x in outs))

#display stats
plt.plot(iters, ltr)
plt.show()

'''
Some nice generated names:
orali.
reglely.
kir.
abellys.
eluna.
szylen.
silyn.
den.
emmalaserrlee.
leon.
marrionty.
cetricollen.
nix.
fairah.
amarquawabie.
tivin.
stensyn.
fyth.
aliza.
joola.
'''