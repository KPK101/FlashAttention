import numpy as np
from scipy.special import softmax as sfmax

d = 1024
n_samples = 10
dataset = np.random.random((d, n_samples))


class attentionModule():
    def __init__(self, d, ffn_expansion=4):
        self.d = d # latent size of input
        self.ffn_expansion = ffn_expansion # expansion factor - hidden layer of MLP
        self.scalefactor = np.sqrt(d)
        self.initParams()

    def initParams(self):
        ## attention head parameters
        self.Wk = np.random.random((d,d))
        self.Wq = np.random.random((d,d))
        self.Wv = np.random.random((d,d))
        ## MLP parameters
        self.Wfc1 = np.random.random((self.d*self.ffn_expansion, self.d))
        self.Wfc2 = np.random.random((self.d, self.d*self.ffn_expansion))
        

    def loadWeights(self,W,type='k'):
        assert(np.shape(W)==(d,d))
        if(type=='k'):
            self.Wk = W
        elif(type=='v'):
            self.Wv = W
        elif(type=='q'):
            self.Wq = W
        elif(type=='fc1')
            assert(W.shape[1]==self.d*self.ffn_expansion and W.shape[0]==self.d)
            self.Wfc1 = W
        elif(type=='fc2'):
            assert(W.shape[0]==self.d*self.ffn_expansion and W.shape[1]=self.d)
            self.Wfc2 = W
        else:
            raise Exception(f"Type has to be {k,v,q} - entered values is {type}")
    
    def softmax(self, x):
        return sfmax(x, axis=0)
    
    def relu(self, x):
        x =  np.maximum(x, 0)
        return x


    def attentionPass(self, x):
        assert(x.shape[0] == d)
        K = self.Wk@x # d*N
        V = self.Wv@x # d*N
        Q = self.Wq@x # d*N

        # compute softmax
        S = self.softmax(K.T@Q) # N*N
        S = S/self.scalefactor
        # compute output
        O = V@S # d*N
        return O
    def mlpPass(self, x):
        x = self.Wfc1@x
        x = self.relu(x)
        x = self.Wfc2@x
        return x

    def forward(self, x):
        x = self.attentionPass(x)
        x = self.mlpPass(x)
        return x

## simple output generation
x = dataset
attmod = attentionModule(d=d)
o = attmod.forward(x)
print(f'input shape = {x.shape}')
print(f'latent size = {attmod.d}')
print(f'output shape = {o.shape}')
        