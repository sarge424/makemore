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
def create_dataset(words):
    X, Y = [], []
    block_size = 3

    for word in words:
        context = [0] * block_size
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

Xtr, Ytr = create_dataset(names[:n2])
#Xdev, Ydev = create_dataset(names[n1:n2])
Xtest, Ytest = create_dataset(names[n2:])


#init parameters
g = torch.Generator().manual_seed(2147483647)

C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True


#gradient descent
ltr = []
iters = []

print('training...')
for i in range(40000):
    #minibatch
    bi = torch.randint(0, Xtr.shape[0], (32,))
    
    #forward pass
    emb = C[Xtr[bi]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[bi])

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    
    #descend
    lr = 0.1
    if i > 30000:
        lr = 0.01
        
    for p in parameters:
        p.data += -lr * p.grad
        
    #stats
    ltr.append(loss.item())
    iters.append(i)
        
    if i % 5000 == 0:
        print('.', i)

#calculate final loss
emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print('training set loss:', loss.item())

emb = C[Xtest]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytest)
print('test set loss:', loss.item())

plt.plot(iters, ltr)
plt.show()