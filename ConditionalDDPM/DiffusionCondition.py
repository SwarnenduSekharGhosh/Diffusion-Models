
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels): # so the lables here is just different from vanilla DDPM because it will act as a condition.
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0) 
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise # In conditional DDPM too we add noise to the target image not to the condition.
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.): # w is the extra parameter here. 
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w
        
        # THIS PART IS THE SAME AS THE VANILLA DDPM
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    # same as vanilla DDPM
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps): # same as vanilla DDPM
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps
    
    def p_mean_variance(self, x_t, t, labels): # only the labels are added wrt the vanilla DDPM
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels) # what noise should i remove to get this class labels.
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device)) # what noise should i remove without class guidance.
        # important :this "nonEps" only works properly if the model was trained to understand label 0 as “no condition”.
        # this usually requires training with random label dropping , sometimes we pass the real label.
        # and sometimes we repalce it with null label. 
        # This is classifier free guidance
        eps = (1. + self.w) * eps - self.w * nonEps # increasing w pushes sample more toward the condition.
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            # create timestep tensor
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # the labels are passed at every reverse step so that every denoising step knows that generate this class.
            mean, var= self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            # add noise except final step
            if time_step > 0: # add nlise except final step.
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            # same DDPM sampling equation but the mean was computed using label-conditioned noise prediction
            # so the image gradually becomes  a sample from the desired class.    
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   # 

