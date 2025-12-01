import torch
import torch.nn.functional as F

# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def timestep_embedding(timesteps, dim):
    """Create sinusoidal timestep embeddings."""
    device = timesteps.device
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return F.pad(emb, (0, 1, 0, 0)) if dim % 2 else emb