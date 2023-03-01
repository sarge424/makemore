import torch

class Bigram:
    def __init__(self, n, bias=0, seed=None):
        self.bias = bias
        self.g = torch.Generator()
        if seed != None:
            self.g.manual_seed(seed)
        
        self.N = torch.zeros((n, n), dtype=torch.int32)
        self.P = self.N.float()
        
    def train(self, xs, ys):
        for xi, yi in zip(xs, ys):
            self.N[xi, yi] += 1
            
        self.P = (self.N + self.bias).float()
        self.P = self.P / self.P.sum(1, keepdim=True)
        
    def __call__(self, x, g):
        return torch.multinomial(self.P[x], 1, replacement=True, generator=g)
    
    #calculate log likelihood - the loss function over training set
    #training set examples are correct, so ideally all the training examples
    #have a probability of 1. Log likelihood is just log of product of probs
    # as log(1) = 0 so its a good loss fn. its also convenient to use
    #log(a*b*c) = loga + logb + logc
    def log_likelihood(self, xs, ys):
        ll = 0.0 #log likelihood
        n = 0  #no of exmaples (to get avg instead of sum)
        for xi,yi in zip(xs, ys):
            prob = self.P[xi, yi]
            logprob = torch.log(prob)
            ll += logprob
            n += 1
            
        return -ll/n