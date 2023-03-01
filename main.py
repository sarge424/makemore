import torch

#read file and get names
names = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(names))))

#define int <-> str mappings
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}

#populate bigram model
N = torch.zeros((27, 27), dtype=torch.int32)
for name in names:
    word = '.' + name + '.'
    for c1,c2 in zip(word, word[1:]):
        i1 = stoi[c1]
        i2 = stoi[c2]
        N[i1, i2] += 1
        
#set deterministic generator
g = torch.Generator().manual_seed(2147483647)

#precalculate probabilities
P = N.float()
P = P / P.sum(1, keepdim=True)

#sample from the model
for _ in range(5):
    outs = []
    i = 0
    while True:
        p = P[i]
        i = torch.multinomial(p, 1, replacement=True, generator=g).item()
        outs.append(itos[i])
        if i == 0:
            break
    print(''.join(outs))
  
#calculate log likelihood - the loss function over training set
#training set examples are correct, so ideally all the training examples
#have a probability of 1. Log likelihood is just log of product of probs
# as log(1) = 0 so its a good loss fn. its also convenient to use
#log(a*b*c) = loga + logb + logc
ll = 0.0 #log likelihood
n = 0  #no of exmaples (to get avg instead of sum)
for name in names:
    word = '.' + name + '.'
    for c1,c2 in zip(word, word[1:]):
        i1 = stoi[c1]
        i2 = stoi[c2]
        prob = P[i1, i2]
        logprob = torch.log(prob)
        ll += logprob
        n += 1
        
print(f'-loglikelihood={-ll/n}')
        
        