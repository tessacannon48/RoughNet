import torch
import math

# =============================================================================
# DIFFUSION SCHEDULERS
# =============================================================================

class LinearDiffusionScheduler:
    """Standard Linear DDPM diffusion scheduler."""
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda', eps=1e-12):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod.clamp(min=0.0, max=1.0))
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt((1 - self.alpha_cumprod).clamp(min=0.0, max=1.0))
        self.alpha_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_cumprod[:-1]], dim=0
        )

        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        ).clamp(min=eps)

        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_cumprod)
        ).clamp(min=eps)

        # Posterior variance and log variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        ).clamp(min=1e-20)
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Reshape the alpha values to match the input tensor shape
        sqrt_alpha_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
    
class CosineDiffusionScheduler:
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021 style).
    Exposes the same attributes as LinearDiffusionScheduler:
      betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod
    """
    def __init__(self, timesteps=1000, s=0.008, device='cuda', eps=1e-12):
        self.timesteps = timesteps
        self.device = device

        # Build alpha_bar (cumprod) with cosine schedule
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        f = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]  

        # Convert alpha_bar -> per-step betas
        alpha_bar = alpha_bar.to(device)
        alpha_bar_t   = alpha_bar[1:]                     
        alpha_bar_t_1 = alpha_bar[:-1].clamp(min=eps)      

        betas = (1.0 - (alpha_bar_t / alpha_bar_t_1)).clamp(1e-8, 0.999)  
        alphas = (1.0 - betas).clamp(min=eps)

        # Store canonical buffers 
        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)                 
        self.alpha_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_cumprod[:-1]], dim=0
        )
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod.clamp(min=0.0, max=1.0))
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(
            (1.0 - self.alpha_cumprod).clamp(min=0.0, max=1.0)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        ).clamp(min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
