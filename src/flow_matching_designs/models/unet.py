from dataclasses import dataclass
from typing import Optional, Sequence
import math
import torch
from torch import nn

from .conditional_vector_field import ConditionalVectorField
from .unet_blocks import ResBlock, Downsample, Upsample
from .registry import register_model


@dataclass
class UNet2DConfig:
    # IO
    in_channels: int = 1          # e.g. 1 for MNIST, 3 for RGB
    out_channels: int = 1         # usually same as in_channels for vector fields
    image_size: Optional[int] = None  # optional sanity check

    # Capacity / structure
    base_channels: int = 64
    channel_multipliers: Sequence[int] = (1, 2, 4, 8)  # scales per resolution
    num_res_blocks: int = 2

    # Conditioning
    num_classes: Optional[int] = 10   # None → no class conditioning (10 real classes for MNIST)
    time_embedding_dim: int = 128
    label_embedding_dim: Optional[int] = None  # if None → use time_embedding_dim


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape (B,) or (B, 1), values in [0, 1]
        returns: shape (B, dim)
        """
        if t.dim() == 2 and t.size(1) == 1:
            t = t[:, 0]
        elif t.dim() != 1:
            raise ValueError(f"t must be (B,) or (B,1), got {t.shape}")

        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(10000.0),
                steps=half_dim,
                device=t.device,
                dtype=t.dtype,
            )
        )
        args = t[:, None] * freqs[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ConditionalUNet2D(ConditionalVectorField):
    """
    General-purpose 2D U-Net with optional time + class conditioning.

    For MNIST:
        cfg = UNet2DConfig(
            in_channels=1,
            out_channels=1,
            num_classes=10,   # real classes; +1 slot internally for null label
            image_size=32,
        )
    """

    def __init__(self, cfg: UNet2DConfig):
        super().__init__()
        self.cfg = cfg

        # ----- Time embedding -----
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.time_embedding_dim),
            nn.Linear(cfg.time_embedding_dim, cfg.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.time_embedding_dim * 4, cfg.time_embedding_dim),
        )

        # ----- Optional label embedding -----
        if cfg.num_classes is not None:
            label_dim = cfg.label_embedding_dim or cfg.time_embedding_dim
            # +1 to reserve an index for the null label used in classifier-free guidance
            self.label_emb = nn.Embedding(cfg.num_classes + 1, label_dim)
            # Project to same dim as time embedding so we can just add them
            if label_dim != cfg.time_embedding_dim:
                self.label_proj = nn.Linear(label_dim, cfg.time_embedding_dim)
            else:
                self.label_proj = nn.Identity()
        else:
            self.label_emb = None
            self.label_proj = None

        # ----- U-Net encoder/decoder -----
        channels = [cfg.base_channels * m for m in cfg.channel_multipliers]

        self.input_conv = nn.Conv2d(cfg.in_channels, channels[0], kernel_size=3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, ch in enumerate(channels):
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(ResBlock(in_ch, ch, emb_dim=cfg.time_embedding_dim))
                in_ch = ch
            if i != len(channels) - 1:
                self.down_blocks.append(Downsample(in_ch))

        # Bottleneck
        self.mid_block1 = ResBlock(in_ch, in_ch, emb_dim=cfg.time_embedding_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, emb_dim=cfg.time_embedding_dim)

        # Up path
        self.up_blocks = nn.ModuleList()
        for i, ch in reversed(list(enumerate(channels))):
            for _ in range(cfg.num_res_blocks):  # symmetric with down path
                self.up_blocks.append(ResBlock(in_ch + ch, ch, emb_dim=cfg.time_embedding_dim))
                in_ch = ch
            if i != 0:
                self.up_blocks.append(Upsample(in_ch))

        self.output_norm = nn.GroupNorm(32, in_ch)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv2d(in_ch, cfg.out_channels, kernel_size=3, padding=1)

    # ------------------------------------------------------------------
    # forward: x (B,C,H,W), t (B,) or (B,1), y (B,) optional
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
            t: (B,) or (B,1) time in [0,1]
            y: (B,) class labels, optional (ignored if cfg.num_classes is None)
        Returns:
            (B, out_channels, H, W)
        """
        if self.cfg.image_size is not None:
            assert x.shape[-1] == self.cfg.image_size and x.shape[-2] == self.cfg.image_size, \
                f"Expected image_size={self.cfg.image_size}, got {x.shape[-2:]}"

        # ---- Build conditioning embedding ----
        temb = self.time_mlp(t)      # (B, time_embedding_dim)
        if self.label_emb is not None and y is not None:
            y_emb = self.label_emb(y)      # (B, label_dim)
            y_emb = self.label_proj(y_emb) # (B, time_embedding_dim)
            temb = temb + y_emb

        # ---- U-Net forward ----
        hs = []
        h = self.input_conv(x)

        # down path
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                h = block(h, temb)
                hs.append(h)
            else:  # Downsample
                h = block(h)

        # mid
        h = self.mid_block1(h, temb)
        h = self.mid_block2(h, temb)

        # up path
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, temb)
            else:  # Upsample
                h = block(h)

        # output
        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)
        return h


@register_model("unet2d")
def build_unet2d(config_dict: dict) -> ConditionalVectorField:
    """
    Factory function to build a ConditionalUNet2D from a raw config dict.
    This is what the registry calls.
    """
    cfg = UNet2DConfig(**config_dict)
    return ConditionalUNet2D(cfg)
