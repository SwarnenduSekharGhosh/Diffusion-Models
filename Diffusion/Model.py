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


# *********** Something worth checking : can we use some other kind of activation here ? *******

# Next what we have is timeembedding. This is important because it tells the Unet model how much 
# noise is present in the current image. Timeembedding converts the integer timestep into a 
# rich feature vector that the U-Net can understand.
 
class TimeEmbedding(nn.Module):
    def __init__(self,T, d_model,dim):
        # T = 1000
        # d_model = initial embedding size
        # dim = final embedding size used inside the U-Net
        assert d_model % 2 ==0
        #later the code creates half sin values half cos values
        super().__init__()
        emb = torch.arange(0, d_model, step = 2) / d_model * math.log(10000)
        # understanding with a small example: 
        # if d_model = 8 then emb = [0,2,4,6]
        # then dividing by 8 we get, emb = [0,0.25,0.50, 0.75]
        # finally multiplying by ln(10000) gives the different freqencies
        emb = torch.exp(-emb)
        # emb now becomes [1, 0.1, 0.01, 0.001] are the frequencies
        # large frequencies capture fine changes
        # small frequencies capture coarse changes
        pos = torch.arange(T).float()
        # For example T = 5
        # then pos = [0 1 2 3 4]
        emb = pos[:, None]*emb[None, :]
        # after this multiplication each row corresponds to one timestep 
        # and column corresponds to one frequency.
        assert list(emb.shape) == [T, d_model//2]
        # checks that the matrix size is correct.
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim =-1)
        # every value becomes sin(value) cos(value)
        # the sine and cosine values are taken to make the timestep integer appear smooth to the network
        assert list(emb.shape) == [T, d_model//2,2]
        # the last dimension saves [sin cos]
        emb = emb.view(T, d_model) # Here the embedding is flattened, 
        
        # creating embedding network
        self.timeembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim), # expands the embedding
            Swish(), # adds non-linearity
            nn.Linear(dim,dim) # allows the network to learn a better representation.
        )
        self.initialize()
    # Initialize weights
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight) #  this keeps activations from exploding or vanishing at the start of training  
                init.zeros_(module.bias) # biases start at zero

    def forward(self,t):
        emb = self.timeembedding(t)
        return emb              
    

"""
Entire workflow of time embedding and why it is important:

1. Timesteps [20, 400, 800]
2. Generate sinusoidal embeddings
3. Embedding lookup table
4. Linear layer
5. Swish activation
6. Linear layer
7. Final time embedding (bach_size, dim)
8. Passed into every ResBlock of the U-Net

Why is this necessary? because the image alone may not tell the network how much denoising is required
The time embedding acts like an instruction:
a. This is step 50 : remove only little noise.
b. Thus is step 900 : remove a lot more noise.

Every diffusion model—DDPM, DDIM, Stable Diffusion, EDM, and many others—includes a time embedding. The specific implementation may vary, 
but the core idea remains the same: convert the timestep into a rich feature vector and inject it throughout 
the U-Net so the network knows where it is in the diffusion process.
"""
# ****** An important question to comeback here : Why do we compute the sinusoidal embeddings ourselves and then store them in an nn.Embedding layer, 
# instead of just computing sin() and cos() inside the forward() method every time?

