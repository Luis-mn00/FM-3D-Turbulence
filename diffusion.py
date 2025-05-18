import numpy as np
import torch
from functools import wraps

def conditional_no_grad(param_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            use_grad = kwargs.get(param_name, False)
            if use_grad:
                return func(*args, **kwargs)
            else:
                with torch.no_grad():
                    return func(*args, **kwargs)
        return wrapper
    
    return decorator

class Diffusion():
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.num_timesteps = config.Diffusion.num_diffusion_timesteps
        
        betas = np.linspace(
            config.Diffusion.beta_start,
            config.Diffusion.beta_end,
            config.Diffusion.num_diffusion_timesteps,
            dtype=np.float64
        )

        self.betas = torch.tensor(betas).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_b = self.alphas.cumprod(axis=0).to(self.device).to(torch.float32)
        self.alphas_b_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_b[:-1]])

        mask_schedule_sig = config.Diffusion.mask_schedule_sig
        mask_schedule = [x**mask_schedule_sig for x in np.linspace(0, 1, self.num_timesteps)]

        mask_schedule = torch.tensor(mask_schedule, dtype=torch.float32)

        self.mask_schedule = mask_schedule.to(self.device)
        
    def forward(self, x_0: torch.Tensor, t: list):
        noise = torch.randn(x_0.shape, device=self.device)
        a = self.alphas_b[t].sqrt().view(-1, 1, 1, 1, 1)
        b = (1 - self.alphas_b[t]).sqrt().view(-1, 1, 1, 1, 1)
        tmp1 = a * x_0
        tmp2 = b * noise
        x_t = tmp1 + tmp2

        # x_t = self.alphas_b[t].sqrt() * x_0 + (1 - self.alphas_b[t]).sqrt() * noise
        return x_t.float(), noise
    
    @conditional_no_grad('grad')
    def ddpm(self, x_inp, model, t_start=None, plot_prog=False, grad=False):
        n = x_inp.size(0)
        t_start = self.num_timesteps if t_start is None else t_start

        x     = x_inp[:, 0:3]
        Re_ch = x_inp[:, 3:4]

        for i in reversed(range(t_start)):
            t = (torch.ones(n) * i).to(x.device)
            b = self.betas[i]
            a = self.alphas[i]
            a_b = self.alphas_b[i]
            model_inp = torch.cat((x, Re_ch), dim=1) 
            e = model(model_inp, t)

            x = (1 / a.sqrt()) * (x - (b / (1 - a_b).sqrt()) * e) 

            if i > 0:
                # x += b.sqrt() * torch.randn_like(x)
                x += torch.randn_like(x) * (b * (1 - self.alphas_b[i - 1]) /(1 - a_b)).sqrt() 

        return x, e # e is optional, delete if errors
    
    @conditional_no_grad('grad')
    def ddpm_mask(self, x, model, x_mask, w_mask=0.3, t_start=None, plot_prog=False, grad=False):
        n = x.size(0)
        t_start = self.num_timesteps if t_start is None else t_start
        mask = torch.rand(x.shape, device=x.device) < w_mask

        for i in reversed(range(t_start)):
            t = (torch.ones(n) * i).to(x.device)
            b = self.betas[i]
            a = self.alphas[i]
            a_b = self.alphas_b[i]
            e = model(x, t)
            
            s_m = self.mask_schedule[i]
            
            # Adding mask of low res values
            x = (1 / a.sqrt()) * (x - (b / (1 - a_b).sqrt()) * e) 

            # x = x * (~mask) + ((1 - s_m) * x + s_m * x_mask) * mask
            
            mask = torch.rand(x.shape, device=x.device) < s_m
            x = x * (~mask) + x_mask * mask

            if i > 0:
                # x += b.sqrt() * torch.randn_like(x)
                x += torch.randn_like(x) * (b * (1 - self.alphas_b[i - 1]) /(1 - a_b)).sqrt() 

        return x


    @conditional_no_grad('grad')
    def ddpm_article(self, x, model, t_start=None, K=1, plot_prog=False, grad=False):
        n = x.size(0)
        t_start = self.num_timesteps if t_start is None else t_start

        for it in range(K):
            t_noise = int(t_start * (0.7 ** it))

            e = torch.randn_like(x)
            x = x * self.alphas_b[t_noise-1].sqrt() + e * (1.0 - self.alphas_b[t_noise-1]).sqrt()

            for i in reversed(range(t_noise)):
                t = (torch.ones(n) * i).to(x.device)
                b = self.betas[i]
                a = self.alphas[i]
                a_b = self.alphas_b[i]
                e = model(x, t)

                x = (1 / a.sqrt()) * (x - (b / (1 - a_b).sqrt()) * e) 

                if i > 0:
                    # x += b.sqrt() * torch.randn_like(x)
                    x += torch.randn_like(x) * (b * (1 - self.alphas_b[i - 1]) /(1 - a_b)).sqrt() 

        return x
    

    @conditional_no_grad('grad')
    def ddim(self, x, model, t_start, reverse_steps, plot_prog=False, grad=False, **kargs):
        seq = range(0, t_start, t_start // reverse_steps) 
        next_seq = [-1] + list(seq[:-1])
        n = x.size(0)
        
        for i, j in zip(reversed(seq), reversed(next_seq)):
            t = (torch.ones(n) * i).to(x.device)
            a_b = self.alphas_b[i]
            a_next_b = self.alphas_b[j] if i > 0 else torch.ones(1, device=x.device)

            e = model(x, t)

            x0_pred = (x - e * (1 - a_b).sqrt()) / a_b.sqrt()
            x = a_next_b.sqrt() * x0_pred + (1 - a_next_b).sqrt() * e

        return x


    @conditional_no_grad('grad')
    def ddim_article(self, x, model, t_start, reverse_steps, K=1, plot_prog=False, grad=False):

        for it in range(K):
            t = int(t_start * (0.7 ** it))
            seq = list(range(0, t, t // min(reverse_steps, t))) 
            t = seq[-1]

            e = torch.randn_like(x)
            x = x * self.alphas_b[t].sqrt() + e * (1.0 - self.alphas_b[t]).sqrt()

            next_seq = [-1] + list(seq[:-1])
            n = x.size(0)
            
            for i, j in zip(reversed(seq), reversed(next_seq)):
                t = (torch.ones(n) * i).to(x.device)
                a_b = self.alphas_b[i]
                a_next_b = self.alphas_b[j] if i > 0 else torch.ones(1, device=x.device)

                e = model(x, t)

                x0_pred = (x - e * (1 - a_b).sqrt()) / a_b.sqrt()
                x = a_next_b.sqrt() * x0_pred + (1 - a_next_b).sqrt() * e

        return x


    @conditional_no_grad('grad')
    def ddim_mask(self, x, model, x_low_res, t_start, reverse_steps, w_mask=0.0, _mask=None, plot_prog=False, grad=False, diff_mask=None, **kargs):
        seq = range(0, t_start, t_start // reverse_steps) 
        next_seq = [-1] + list(seq[:-1])
        n = x.size(0)
        
        mask = torch.rand(x.shape, device=x.device) < w_mask

        if _mask is not None:
            mask = torch.clamp(mask + _mask, max=1)

        if diff_mask is not None:
            mask = diff_mask

        for i, j in zip(reversed(seq), reversed(next_seq)):
            t = (torch.ones(n) * i).to(x.device)
            a_b = self.alphas_b[i]
            a_next_b = self.alphas_b[j] if i > 0 else torch.ones(1, device=x.device)
            e = model(x, t)
            x0_pred = (x - e * (1 - a_b).sqrt()) / a_b.sqrt()

            mask_t = mask * self.mask_schedule[i]
            x_masked = x0_pred * (1 - mask_t) + x_low_res * mask_t

            x = a_next_b.sqrt() * x_masked + (1 - a_next_b).sqrt() * e
                
        return x, mask