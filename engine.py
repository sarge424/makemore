import torch

class Bigram:
    def __init__(self, n, _bias=0):
        self.bias = _bias
        
        self.N = torch.zeros((n, n), dtype=torch.int32)
        self.P = self.N.float()
        
    def train(self, xs, ys):
        for xi, yi in zip(xs, ys):
            self.N[xi, yi] += 1
            
        self.P = (self.N + self.bias).float()
        self.P = self.P / self.P.sum(1, keepdim=True)
        
    def sample(self, x, g):
        return torch.multinomial(self.P[x], 1, replacement=True, generator=g)