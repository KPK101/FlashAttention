import numpy as np
from scipy.special import softmax as sfmax

d = 1024
n_samples = 10
dataset = np.random.random((d, n_samples))


class attentionModule():
    def __init__(self, d):
        self.d = d # latent size of input
        self.initParams()

    def initParams(self):
        self.Wk = np.random.random((d,d))
        self.Wq = np.random.random((d,d))
        self.Wv = np.random.random((d,d))

    def loadWeights(self,W,type='k'):
        assert(np.shape(W)==(d,d))
        if(type=='k'):
            self.Wk = W
        elif(type=='v'):
            self.Wv = W
        elif(type=='q'):
            self.Wq = W
        else:
            raise Exception(f"Type has to be {k,v,q} - entered values is {type}")
    
    def softmax(self, x):
        return sfmax(x, axis=0)

    def mlp(self,x):
        pass
        
    def attentionPass(self, x):
        assert(x.shape[0] == d)
        K = self.Wk@x # d*N
        V = self.Wv@x # d*N
        Q = self.Wq@x # d*N

        # compute softmax
        S = self.softmax(K.T@Q) # N*N
        # compute output
        O = V@S # d*N
        return O

    def forward(self, x):
        O = self.attentionPass(x)
        return O

## simple output generation
x = dataset
am = attentionModule(d=d)
O = am.forward(x)
print(f'input shape = {x.shape}')
print(f'latent size = {am.d}')
print(f'output shape = {O.shape}')
        