import torch
from engine import Bigram

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
        
        