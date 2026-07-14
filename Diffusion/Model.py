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
# look at every other pixel/location and decide what information is important. Normal convolution mostly looks
# locally, but attention can look globally.
# attention = softmax(q x k)
# output = attention x v
# finally return x + h , this is a residula connection

class AttnBlock(nn.Module):
    def __init__(self, in_ch): 
        super().__init__()
        self.group_norm = nn.GroupNorm(32,in_ch) # in_ch must be divisible by 32
        # although it starts from same feature map but independent linear transformations 
        # so that they donot share weights 
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch,in_ch,1 , stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        # why all of them are 1x1 convolution ?
        # because attention already mixes spatial information.The convolution doesnot need to look at neighbouring pixels.
        # Its only job is to transform the channel representation.
        self.initialize()
    
    # Initialize every convolution with standard Xavier initilization
    # it prevent exploding activations
    # and, vanishing activations at the beginning of training.
    def initialize(self):
        for module in [self.proj_q, self.proj_k,self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias) #starts every bias at zero
        
        init.xavier_uniform_(self.proj.weight, gain=1e-5)
    # Important : 
    def forward(self,x):
        B,C,H,W = x.shape

        h  = self.group_norm(x)  # normalize all the feature map before attention

        # create q, k , v
        q = self.proj_q(h)    
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0,2,3,1).view(B, H*W, C)
        k = k.view(B,C,H*W)

        w = torch.bmm(q,k)*(int(C) ** (-0.5)) # batch matrix multiplication (B, H*W,C) X (B,C,H*W) = (B, H*W, H*W)
                                              # the scaling term c **(-0.5) keeps the attention scores stable
        assert list(w.shape) == [B, H * W, H * W] # to check the final dimension of w
        w = F.softmax(w, dim=-1) # compute attention scores
        
        
        v = v.permute(0,2,3,1).view(B,H*W,C) # reshape "value"

        h = torch.bmm(w,v) # batch matrix multiplication (every location receives information from all other locations)
        assert list(h.shape) == [B,H*W,C]  # to check the final dimension of h
        h = h.view(B,H,W,C).permute(0,3,1,2) # Here, (B, H*W, C) becomes (B,C,H,W) so it is again normal image-like feature map
        h = self.proj(h) # another 1x1 convolution. This mixes the attended channel information

        return x + h   # original feature map + attention-modified feature map
                       # this doesnot completely replace "x", but it adds useful global information to it.
                        
""" ResBlock : this block helps the U-Net process the noisy image while knowing the current diffusionU-Net.

Input feature x
      │
      ▼
Normalize + activation + convolution
      │
      ▼
Add time embedding information
      │
      ▼
Normalize + activation + dropout + convolution
      │
      ▼
Add shortcut connection
      │
      ▼
Optional attention
      │
      ▼
Output feature


"""
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn = False):
        # in_ch   = number of input channels (128)
        # out_ch  = number of output channels (256)
        # tdim    = dimension of time embedding (512)
        # drpout  = dropout probability
        # attn    = whether to use attention or not 
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32,in_ch), #the in_ch must be divisible by 32
            Swish(), # this gives smooth non-linearity.
            nn.Conv2d(in_ch,out_ch, 3, stride = 1, padding = 1), #this changes the number of channels.  
        )
        # time embedding projection
        # This part takes the time embedding and converts it to the 
        # same number of channels as the feature map.
        self.temb_proj = nn.Sequential( 
            Swish(),
            nn.Linear(tdim, out_ch), # temb_proj(temb).shape = (4, 256)
        )
        # Because we need to add this time information to the feature map
        # whose channel dimension is now out_ch
        # the next block continues processing after the time embedding has been added.
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout), # roughly dropout% of activations are dropped.
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),    
        ) # This keeps the number of channels the same

        if in_ch !=out_ch: #this is the residual connection
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1 , stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        # self attention is created here.   
        if attn:
                self.attn = AttnBlock(out_ch)
        else:
                self.attn = nn.Identity()
          
        self.initialize() 
    
    # This initializes all convolution and linear layers.
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight) # initializes the weights in a stable way.
                init.zeros_(module.bias) # setting bias to zero.
        init.xavier_uniform_(self.block2[-1].weight, gain = 1e-5)  # this re-initializes the last convolution
        # block 2  with a very small gain.     
        # why make it so small?
        # because later we see the ResBlock output is h = h + self.shortcut(x)
        # at the beginning of training we want, h = 0,
        # output - shortcut(x), this means the ResBlock initially behaves almost like an identity mapping.

    def forward(self, x , temb):
            h = self.block1(x)
            """ This does : 
            GroupNorm     (4, 128, 32, 32)
              ↓
            Swish
              ↓
            Conv2d        (4, 256, 32, 32)
            """
            h+= self.temb_proj(temb)[:,:, None, None]
            """
            This step is very important : as the Pytorch broadcasts the time information across height and width to be added to 
            every spatial location.
            """
            h = self.block2(h)
            """
            the second convolution block. This does,
            GroupNorm
            ↓
            Swish
            ↓
            Dropout
            ↓
            Conv2d
            """
            
            h = h + self.shortcut(x)
            """
            so the block adds the transformed feature 'h' to the original input x.
            the shape of 'h' and shape of x are different so cannot be added directly.
            so the shortcut uses a 1x1 convolution

            The block learns a correction to the input, rather than learning the whole transformation.
            """

            
            h = self.attn(h)
            """
            if attn = True, then the feature map goes through self-attention.
            if attn = False, which means nothing changes.
            """
            return h
                
"""

we can think of the ResBlock like this: 
Input feature x
      │
      ├─────────────── shortcut path ───────────────┐
      │                                             │
      ▼                                             │
GroupNorm → Swish → Conv                            │
      │                                             │
      ▼                                             │
Add time embedding                                  │
      │                                             │
      ▼                                             │
GroupNorm → Swish → Dropout → Conv                  │
      │                                             │
      └──────────── add shortcut(x) ◄───────────────┘
                    │
                    ▼
              Optional Attention
                    │
                    ▼
                 Output

"""
class UNet(nn.Module):
    def __init(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i< len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        
        self.head = nn.Conv2d(3, ch, kernel_size = 3, stride = 1, padding = 1)
        self.downblocks = nn.ModuleList()
        chs = [ch] #record output channel when downsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch*mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch = now_ch, out_ch = out_ch, tdim = tdim,
                    dropout = dropout, attn = (i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i !=len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn = True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn = False),
        ])        
        
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch = chs.pop() + now_ch, out_ch = out_ch, tdim = tdim,
                    dropout = dropout, attn = (i in attn)))
                now_ch = out_ch
            if i in !=0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0 


        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride = 1, padding = 1)
        )             
        self.initialize()
        
