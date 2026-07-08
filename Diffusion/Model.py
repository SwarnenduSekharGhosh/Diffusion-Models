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
        # later the code creates half sin values half cos values
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


# **************** Now we start to code the structure of the U-Net *************

"""
1. Input image
2. ResBlock
3. DownSample     # reduce the spatial size if the feature maps
4. ResBlock
5. DownSample     # reduce the spatial size if the feature maps
6. Bottleneck
7. UpSample
8. ResBlock
9. UpSample
10. Output

"""
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride = 2, padding = 1) #this is one convolution
        self.initialize()                      # with stride = 2 the output becomes smaller

    def initialize(self):
        init.xavier_uniform_(self.main.weight) # initializes the convolution weights. 
        init.zeros_(self.main.bias) # makes every bias zero.

    def forward(self,x,temb):
        x = self.main(x)
        # The convolution performs
        # Learn filters.
        # Reduce resolution.
        # Produce the next feature map.
        return x
    
## Changing the number of channels is usually done inside the ResBlock, not the DownSample block.    
    
"""
An important note : 

Why not use MaxPool2d ?

Early CNNs like AlexNet and VGG often used max pooling to reduce the spatial dimensions. 
Modern architectures-Including ResNet, many U-nets and most diffusion models prefer stride 2- convolutions
because they are learnable.
A max-pooling layer always performs the same fixed operation. A stride 2- convolution learns how to downsample
in the most useful way for the task. In diffusion models, preserving important image details during
the encoder path is crucial for accurate denoising so a learnable downsampling operation is generally preferred.
"""
        
class UpSample(nn.Module): #the upsample block expands it back to the original resolution
    def __init__(self, in_ch):
        super().__init__()  
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        # unlike downsample this convolution doesnot change the image size
        # it only learns better features after the image has been enlarged
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight) # initializes the conv weights
        init.zeros_(self.main.bias)       # makes every bias zero
    
    def forward(self,x,temb):
        _,_,H,W = x.shape
        x = F.interpolate(
            x, scale_factor = 2, mode='nearest') # the original feature map is upscaled (doubled)
                               #nearest neighbour interpolation simply copies every pixel. nothing is learned yet
        x = self.main(x) # after enlarging the feature map, the convolution learns how to refine it.
        return x         # the convolution learns meaningful features after upsampling

"""
These two blocks form the "U" shape of the U-Net. The encoder repeatedly DownSamples to capture 
increasingly abstract information,  and the decoder repeatedly UpSamples while combining 
features from the encoder
"""

# This is the self attention block of the diffusion U-Net : the job is to let every pixel/location
# look at every other pixel/location and decide what information is important.

class AttnBlock(nn.Module):
    def __init__(self, in_ch): 
        super().__init__()
        self.group_norm = nn.GroupNorm(32,in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch,in_ch,1 , stride=1, padding=0)
        self.proj = nn. Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k,self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self,x):
        B,C,H,W = x.shape
        h  = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0,2,3,1).view(B, H*W, C)
        k = k.view(B,C,H*W)
        w = torch.bmm(q,k)*(int(C) ** (-0.5))
        assert list(w.shape) == [B,H * W,H * W]
        W = F.softmax(w, dim=-1)
        

        v = v.permute(0,2,3,1).view(B,H*W,C)
        h = torch.bmm(w,v)
        assert list(h.shape) == [B,H*W,C]
        h = h.view(B,H,W,C).permute(0,3,1,2)
        h = self.proj(h)

        return x + h

