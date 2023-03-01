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

#generate words
for _ in range(20):
    outs = []
    i = 0
    while True:
        p = N[i].float()
        p = p / p.sum()
        i = torch.multinomial(p, 1, replacement=True, generator=g).item()
        outs.append(itos[i])
        if i == 0:
            break
    print(''.join(outs))