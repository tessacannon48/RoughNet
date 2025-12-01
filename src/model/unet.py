import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, Down, Up, SelfAttention2D
from src.diffusion.utils import timestep_embedding

# =============================================================================
# PER-VIEW ATTRIBUTE-AWARE SPATIAL POOLING MODULE
# =============================================================================

class AttrAwareSpatialPool(nn.Module):
    """
    Attribute-aware spatial pooling over a set of k Sentinel-2 views.

    For each LiDAR patch, we have:
      - k Sentinel-2 patches (4 bands each)
      - k attribute vectors 

    The module computes a spatially varying attention weight over the k views using
    both image features and attributes, and returns a single fused feature map.
    """
    def __init__(self, in_ch=4, feat_ch=32, attr_dim=8, hid_attr=32, score_hid=32, temperature_init=1.0):
        super().__init__()
        # 1) Shared per-view image encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, feat_ch, 3, padding=1), nn.SiLU(),
        )
        # 2) Per-view attribute encoder -> project attrs to feat_ch
        self.attr_mlp = nn.Sequential(
            nn.Linear(attr_dim, hid_attr), nn.SiLU(),
            nn.Linear(hid_attr, feat_ch), nn.SiLU(),
        )
        # 3) Score head over views (spatially varying weights)
        self.score_head = nn.Sequential(
            nn.Conv2d(2*feat_ch, score_hid, 1), nn.SiLU(),
            nn.Conv2d(score_hid, 1, 1)
        )
        # Learnable softmax temperature (log-space parameterization)
        self.log_temp = nn.Parameter(torch.tensor(float(temperature_init)).log())

    def forward(self, cond_img, attrs_kA, k):
        """
        Args:
            cond_img : torch.Tensor
                Concatenated Sentinel-2 patches for each LiDAR patch.
                Shape: [B, k*4, H, W] where each group of 4 channels is one view.
            attrs_kA : torch.Tensor
                Per-view attributes (e.g., cloud, sun/view angles, age).
                Shape: [B, k, A], where A typically = 8.
            k : int
                Number of views (Sentinel-2 patches) per LiDAR patch.

        Returns:
            fused : torch.Tensor
                Fused feature map after attribute-aware pooling across views.
                Shape: [B, C, H, W], where C = feat_ch.
        """
        B, _, H, W = cond_img.shape
        # Split concatenated channels into k views: [B, k, 4, H, W]
        views = cond_img.view(B, k, 4, H, W)

        # Encode each view with the shared CNN encoder
        feats = []
        for i in range(k):
            feats.append(self.enc(views[:, i]))          # [B, C, H, W]
        Fv = torch.stack(feats, dim=1)                   # [B, k, C, H, W]

        # Encode attributes and broadcast to spatial maps
        Aenc = self.attr_mlp(attrs_kA)                   # [B, k, C]
        Amap = Aenc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)  # [B, k, C, H, W]

        # Compute per-view, per-pixel scores
        Z = torch.cat([Fv, Amap], dim=2)                 # [B, k, 2C, H, W]
        S = self.score_head(Z.view(B * k, -1, H, W))     # [B*k, 1, H, W]
        S = S.view(B, k, 1, H, W).squeeze(2)             # [B, k, H, W]

        # Softmax over views with learnable temperature
        temp = self.log_temp.exp().clamp(1e-3, 100.0)
        Wght = torch.softmax(S / temp, dim=1)            # [B, k, H, W]

        # Weighted sum over views -> fused feature map
        fused = (Wght.unsqueeze(2) * Fv).sum(dim=1)      # [B, C, H, W]
        return fused


# =============================================================================
# CONDITIONAL U-NET WITH ASAP FUSION
# =============================================================================

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for diffusion on LiDAR residuals with multi-view Sentinel-2 conditioning.

    - Input:
        x        : noisy LiDAR RANSAC residuals, shape [B, 1, H, W]
        cond_img : k Sentinel-2 patches (R,G,B,NIR) concatenated along channels,
                   shape [B, k*4, Hc, Wc]
        attrs    : flattened per-view attributes, shape [B, k*8]
        t        : diffusion timesteps, shape [B]

    - Conditioning:
        * Pixel-level: fused Sentinel-2 features from AttrAwareSpatialPool
        * Global: time embedding (t) broadcast into all U-Net blocks via FiLM-style
                  conditioning (handled inside DoubleConv / Down / Up blocks).
    """
    def __init__(self, in_channels=1, cond_channels=24, attr_dim=48, base_channels=128, embed_dim=256, unet_depth=4, attention_variant='default', cond_k=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_variant = attention_variant
        self.base_channels = base_channels
        self.cond_k = cond_k
        self.attr_dim = attr_dim
        self.attr_dim_per = self.attr_dim // self.cond_k  # should be 8

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Permutation-invariant fusion of k Sentinel-2 views (ASAP)
        self.asap = AttrAwareSpatialPool(
            in_ch=4,
            feat_ch=cond_channels,
            attr_dim=self.attr_dim_per,
        )

        # U-Net backbone
        # Input convolution: LiDAR + fused Sentinel-2 features
        self.input_conv = DoubleConv(in_channels + cond_channels, base_channels, embed_dim)

        # Encoder path with downsampling blocks
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i in range(unet_depth):
            max_channels = base_channels * 8
            out_ch = min(in_ch * 2, max_channels)
            use_attn = self._use_attention(i, stage='down', depth=unet_depth)
            self.downs.append(Down(in_ch, out_ch, embed_dim, use_attention=use_attn))
            in_ch = out_ch

        # Bottleneck block at the lowest resolution
        self.bottleneck_conv = DoubleConv(in_ch, in_ch, embed_dim, use_attention=True)

        # Decoder path with upsampling blocks
        self.ups = nn.ModuleList()
        for i in range(unet_depth):
            in_ch_prev = in_ch
            # Channels from the corresponding encoder skip connection
            skip_ch = self._get_skip_channels(unet_depth, i)
            out_ch = skip_ch
            use_attn = self._use_attention(i, stage='up', depth=unet_depth)
            self.ups.append(Up(in_channels=in_ch_prev, skip_channels=skip_ch, out_channels=out_ch, embed_dim=embed_dim, use_attention=use_attn))
            in_ch = out_ch

        # Final projection back to LiDAR channel dimension
        self.output_conv = nn.Conv2d(in_ch, in_channels, 1)


    def _get_skip_channels(self, depth, current_up_idx):
        """
        Compute the number of channels in the skip connection corresponding
        to a given decoder block index, assuming each Down block doubles channels
        up to a maximum of 8 × base_channels.
        """
        channels = [self.base_channels]
        for i in range(depth):
            max_channels = self.base_channels * 8
            out_ch = min(channels[-1] * 2, max_channels)
            channels.append(out_ch)
        # We traverse decoder blocks from deepest to shallowest; index backwards.
        return channels[-(current_up_idx + 2)]

    def _use_attention(self, idx, stage, depth):
        """
        Decide whether to enable self-attention in a given encoder/decoder block.

        Args:
            idx   : 0-based index of the block within its stage
            stage : 'down' for encoder, 'up' for decoder
            depth : total number of encoder/decoder blocks (excluding bottleneck)

        Variants:
            - 'none'   : no attention
            - 'all'    : attention in every block
            - 'mid'    : attention only in the innermost encoder & first decoder block
            - 'heavy'  : like 'mid', plus one neighbouring block on each side:
                         encoder idx in {depth-2, depth-1}, decoder idx in {0, 1}
            - 'default': currently same as 'none' (no attention)
        """
        if self.attention_variant == 'none':
            return False
        elif self.attention_variant == 'all':
            return True
        elif self.attention_variant == 'mid':
            if stage == 'down' and idx == depth - 1:
                return True
            if stage == 'up' and idx == 0:
                return True
            return False
        elif self.attention_variant == 'heavy':
            if depth >= 2:
                if stage == 'down' and idx >= depth - 2:  
                    return True
                if stage == 'up' and idx <= 1:            
                    return True
                return False
            else:
                # Fallback to 'mid' when there is only one block
                if stage == 'down' and idx == depth - 1:
                    return True
                if stage == 'up' and idx == 0:
                    return True
                return False
        elif self.attention_variant == 'default':
            return False
        else:
            return False
        
    def forward(self, x, cond_img, attrs, t):
        """
        Forward pass of the conditional U-Net.

        Args:
            x : torch.Tensor
                Noisy LiDAR RANSAC residuals.
                Shape: [B, 1, H, W]
            cond_img : torch.Tensor
                Sentinel-2 conditioning patches (k views × 4 bands).
                Shape: [B, k*4, Hc, Wc]
            attrs : torch.Tensor
                Flattened per-view attributes for the k Sentinel-2 patches.
                Shape: [B, k*8]
            t : torch.Tensor
                Diffusion timesteps.
                Shape: [B]

        Returns:
            torch.Tensor of shape [B, 1, H, W]: predicted noise / residuals.
        """

        B = x.size(0)
        k = self.cond_k
        A_per = self.attr_dim_per
    
        # Time embedding 
        t_emb = self.time_mlp(timestep_embedding(t, self.embed_dim))

        # Fuse S2 views with attribute-aware spatial pooling (ASAP)
        attrs_kA = attrs.view(B, k, A_per)
        fused_cond = self.asap(cond_img, attrs_kA, k=k)

        # Concatenate LiDAR input and fused Sentinel-2 features
        x = torch.cat([x, fused_cond], dim=1)

        # Encoder path: downsample and save skip connections
        skips = []
        x = self.input_conv(x, t_emb)
        for down in self.downs:
            skips.append(x)
            x = down(x, t_emb)

        # Bottleneck at lowest spatial resolution
        x = self.bottleneck_conv(x, t_emb)

        # Decoder path: upsample and fuse with corresponding skips
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb)

        # Final projection back to single-channel LiDAR residual space
        return self.output_conv(x)
