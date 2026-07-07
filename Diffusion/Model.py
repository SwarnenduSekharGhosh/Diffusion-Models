import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Swish(nn.Module):             # Here the swish activation function has been defined 
    def forward(self,x):     
        return x * torch.signoid(x)
    
# Mathematically it means: 
# Swish(x) = x . sigma(x)
# where sigma(x) = 1/(1+exp(-x))
# Hence, Swish(x) = x . 1/(1+exp(-x))

# In neural networks after a linear layer or convolution layer we apply a non-linear 
# function which is called as the activation function. Swish is one of those activation functions
# that takes an x and multiplies it with sigmoid(x). Unlike ReLu that completely kills the 
# negative values, Swish keeps a little bit of the negative information (negative value). 
# So Swish is smoother.Diffusion models use UNet like architectures. These networks need stable and smooth
# transformations because they predict noise at many different timesteps. 


# *********** Something worth checking can we use some other kind of activation here *******

class TimeEmbedding(nn.Module):
    def __init__(self,T, d_model,dim):
        assert d_model % 2 ==0
        super().__init__()
        emb = torch.arange(0, d_model, step = 2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None]*emb[None, :]
        assert list(emb.shape) == [T, d_model//2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim =-1)
        assert list(emb.shape) == [T, d_model//2,2]
        emb = emb.view(T, d_model)

        self.timeembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb)
            nn.Linear(d_model, dim)
            Swish(),
            nn.Linear(dim,dim)
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)   
                init.zeros_(module.bias)

    def forward(self,t):
        emb = self.timeembedding(t)
        return emb              