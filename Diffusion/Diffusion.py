import torch 
import torch.nn as nn
import torch.nn. functional as F

import numpy as np

def extract(v,t,x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1,1,1,...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v,index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape)-1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
                'betas', torch.linspace(beta_1, beta_T, T).double()) # 'beta' tells how much noise is added
        alphas = 1. - self.betas # alpha tells hw much of the original image remains
        alphas_bar = torch.cumprod(alphas,dim=0)  # After many diffusion steps how much of original signal is left

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
          'sqrt_alphas_bar', torch.sqrt(alphas_bar)) 
        self.register_buffer(
            'sqrt_one_minus_alphas_bar',torch.sqrt(1. - alphas_bar)) 
                    
    # The diffusion equation is : x_t = sqrt_alphas_bar*x_0 + sqrt_one_minus_alphas_bar*noise 
    def forward(self,x_0):
        """
        Algorithm 1.
        """    
        t = torch.randint(self.T, size = (x_0.shape[0],)),device=x_0.device
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar,t,x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t,t),noise, reduction='none')
        return loss
        # The training minimizes the error between the actual noise and the model(preferrably U-Net predicted Noise)
 
        """
        In short, every training iteration follows this sequence:

        1. Start with a clean image x_0.
        2. Randomly choose a timestep t.
        3. Sample Gaussian noise ϵ.
        4. Create a noisy image x_t
	​    5. using the closed-form diffusion equation.
        6. Feed (x_t,t) to the model.
        7. Compute the MSE between the predicted noise and the true noise.
        8. Update the model so it gets better at predicting the added noise. This is the core learning objective of DDPM training.
        """

class GaussianDiffusionSampler(nn.Module):
    def __init__(self,model,beta_1,beta_T,T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas',torch.linspace(beta_1, beta_T,T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim = 0)

        alphas_bar_prev = F.pad(alphas_bar,[1,0], value=1)[:T] 
        # This over here is a little trick
        # For example if alpha_bar = [0.99, 0.98, 0.96, 0.94]
        # Padding adds a 1 at the beginning
        # then, alpha_bar = [1, 0.99, 0.98, 0.96, 0.94]
        # After that [:T] keeps only [1, 0.99, 0.98, 0.96]
        # so now alphas_bar_prev[t] = alphas_bar[t-1]


        # now we get the variables from the DDPM derivation
        self.register_buffer('coeff1', torch.sqrt(1./ alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas*(1. - alphas_bar_prev)/(1. - alphas_bar))


    # predict previous mean
    def predict_xt_prev_mean_from_eps(self,x_t,t, eps):
        # x_t is the current noisy image
        # eps is the predicted noise
        # returns mean of x_{t-1}
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape)*x_t- # this is the exact DDPM equation
            extract(self.coeff2, t, x_t.shape)*eps
        )    
    # The network predicts the noise, and this formula converts that prediction into the mean of the previous timestep.

    def p_mean_variance(self,x_t,t): # its job is to compute the Gaussian distribution
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2],self.betas[1:]])
        # The first element is never actually used in sampling because at t=0 no noise is added.
        var = extract(var,t,x_t.shape)
        # selects the correct variance for each image in the batch.

        eps = self.model(x_t,t) # The U-Net predicts the noise not an image
        
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t,t,eps=eps)
        
        return xt_prev_mean, var 
        # Hence we now have the mean and the variance p(x_{t-1}|x_t)
    
    def forward(self, x_T): 
        """
        Algorithm 2.
        """

        x_t = x_T
        
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0],],dtype=torch.long)*time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t==0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0 # no more randomness
            # otherwise the final image would still contain noise    

            x_t = mean + torch.sqrt(var) * noise #  we move one step closer to clean image
            # NaN check
            assert torch.isnan(x_t).int().sum() == 0 , "nan in tensor." # making sure no invalid numbers appear

        x_0 = x_t
        return torch.clip(x_0, -1, 1)       
    # images were trained in the range [-1,1], so the output is clipeed to that range.  

"""
GaussianDiffusionTrainer learns a model that predicts the noise ϵ added to an image at any timestep.
GaussianDiffusionSampler uses that trained model to iteratively estimate and remove noise, one timestep at a time, starting from pure Gaussian noise.

So the trainer teaches the U-Net how to recognize noise, and the sampler uses that knowledge to turn random noise into a realistic image.
""" 
          

