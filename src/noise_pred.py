import torch
import re
import torch.nn as nn
from collections import OrderedDict

class HookBank:
    """Store intermediate outputs collected from forward hooks."""
    def __init__(self):
        self.buffers = OrderedDict()
        self.hooks = []

    def add(self, module, name, proj=None):
        def _hook(_, __, out):
            y = out if proj is None else proj(out)
            self.buffers[name] = y
        h = module.register_forward_hook(_hook)
        self.hooks.append(h)

    def clear(self):
        self.buffers.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class FuseHead(nn.Module):
    """Fuse multi-scale feature maps into a shared latent representation."""
    def __init__(self, in_chs, out_ch_latent=128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(sum(in_chs), out_ch_latent, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(out_ch_latent, out_ch_latent, 3, 1, 1),
            nn.SiLU(),
        )

    def forward(self, feats_to_merge, target_hw):
        """
        Args:
            feats_to_merge: List of feature maps with shape [B, C_i, h_i, w_i].
            target_hw: Target spatial size (Ht, Wt).

        Returns:
            Fused latent feature map with shape [B, C_latent, Ht, Wt].
        """
        xs = []
        Ht, Wt = target_hw
        for x in feats_to_merge:
            h, w = x.shape[-2:]
            # Resize each feature map to the target resolution before concatenation.
            if h > Ht or w > Wt:
                kh, kw = max(1, h // Ht), max(1, w // Wt)
                x = nn.functional.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw), ceil_mode=True)
            elif h < Ht or w < Wt:
                x = nn.functional.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        return self.fuse(x)  # [B, C_lat, Ht, Wt]

class TimestepHead(nn.Module):
    """Predict a scalar/logit from the fused latent feature, optionally with text features."""
    def __init__(self, c_lat, txt_dim=0):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.LayerNorm(c_lat+txt_dim),
            nn.Linear(c_lat+txt_dim, 256), nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)   # For regression or binary logit prediction
        )
    def forward(self, x_lat, txt=None):
        v = self.gap(x_lat).flatten(1)
        if txt is not None:
            v = torch.cat([v, txt], dim=1)
        return self.fc(v)

class EpsHead(nn.Module):
    """Predict residual/noise-like outputs from the latent feature map."""

    def __init__(self, c_lat, out_ch=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(c_lat, 64, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(64, out_ch, 3, 1, 1),
        )
    def forward(self, x_lat):
        """
        Args:
            x_lat: Latent feature map [B, C, H, W].

        Returns:
            Output tensor with shape [B, out_ch, H, W].
        """
        return self.head(x_lat)
