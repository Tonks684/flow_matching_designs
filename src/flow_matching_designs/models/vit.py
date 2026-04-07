"""
Vision Transformer (ViT) as a conditional vector field for flow matching.

Architecture: patchify → transformer blocks (AdaLN-Zero conditioning) → unpatchify.
Time and class conditioning are injected via Adaptive Layer Norm (AdaLN-Zero),
following the DiT design (Peebles & Xie, 2023).

For MNIST at 32×32 with patch_size=4:
    - 64 patches of shape (4, 4, 1) embedded to hidden_dim tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .conditional_vector_field import ConditionalVectorField
from .registry import register_model


# ---------------------------------------------------------------------------
# Time embedding (sinusoidal, same as UNet)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) or (B, 1) in [0, 1] → (B, dim)"""
        if t.dim() == 2 and t.size(1) == 1:
            t = t[:, 0]
        elif t.dim() != 1:
            raise ValueError(f"t must be (B,) or (B,1), got {t.shape}")
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), steps=half_dim,
                           device=t.device, dtype=t.dtype)
        )
        args = t[:, None] * freqs[None, :]          # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ViT2DConfig:
    # IO
    in_channels: int = 1
    out_channels: int = 1
    image_size: int = 32

    # Patch embedding
    patch_size: int = 4             # image_size must be divisible by patch_size

    # Transformer capacity
    hidden_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Conditioning
    num_classes: Optional[int] = 10  # None → unconditional; +1 slot reserved for null label
    time_embedding_dim: int = 128
    label_embedding_dim: Optional[int] = None  # defaults to hidden_dim

    # Image conditioning (channel concatenation)
    cond_channels: int = 0  # set > 0 to enable image-to-image conditioning


# ---------------------------------------------------------------------------
# Transformer block with AdaLN-Zero conditioning
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with Adaptive Layer Norm (AdaLN-Zero).

    The conditioning embedding modulates scale and shift of each LayerNorm
    and gates the residual connections.  The modulation projection is
    zero-initialised so the model starts as an identity map (DiT §3.2).
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float,
                 emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        # Projects conditioning to (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 6 * hidden_dim, bias=True),
        )
        # Zero-init → identity at initialisation
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, N, hidden_dim)
        emb: (B, emb_dim)
        """
        # Six modulation parameters, each (B, 1, hidden_dim) for broadcasting over N
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            v.unsqueeze(1)
            for v in self.adaLN_modulation(emb).chunk(6, dim=-1)
        )

        # Self-attention branch
        h = self.norm1(x) * (1 + scale1) + shift1
        attn_out, _ = self.attn(h, h, h)
        x = x + gate1 * attn_out

        # MLP branch
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# ViT2D — the full model
# ---------------------------------------------------------------------------

class ViT2D(ConditionalVectorField):
    """
    Vision Transformer conditional vector field for flow matching.

    forward(x, t, y) → vector field estimate with same shape as x.
    """

    def __init__(self, cfg: ViT2DConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.image_size % cfg.patch_size == 0, (
            f"image_size {cfg.image_size} must be divisible by patch_size {cfg.patch_size}"
        )

        grid_size = cfg.image_size // cfg.patch_size
        self.num_patches = grid_size * grid_size

        # ----- Patch embedding -----
        # Equivalent to flattening patches and projecting; a single strided Conv2d is cleaner.
        # When cond_channels > 0, condition image is channel-concatenated before embedding.
        self.patch_embed = nn.Conv2d(
            cfg.in_channels + cfg.cond_channels, cfg.hidden_dim,
            kernel_size=cfg.patch_size, stride=cfg.patch_size,
        )

        # ----- Learnable positional embedding -----
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, cfg.hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # ----- Time conditioning -----
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.time_embedding_dim),
            nn.Linear(cfg.time_embedding_dim, cfg.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.time_embedding_dim * 4, cfg.hidden_dim),
        )

        # ----- Class conditioning -----
        if cfg.num_classes is not None:
            label_dim = cfg.label_embedding_dim or cfg.hidden_dim
            # +1 slot for the null label used in classifier-free guidance
            self.label_emb = nn.Embedding(cfg.num_classes + 1, label_dim)
            self.label_proj = (
                nn.Linear(label_dim, cfg.hidden_dim)
                if label_dim != cfg.hidden_dim
                else nn.Identity()
            )
        else:
            self.label_emb = None
            self.label_proj = None

        # ----- Transformer blocks -----
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                emb_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])

        # ----- Output head -----
        self.out_norm = nn.LayerNorm(cfg.hidden_dim, eps=1e-6)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_channels * cfg.patch_size ** 2)
        # Zero-init for stable early training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    # Patchify / unpatchify helpers
    # ------------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, hidden_dim)"""
        h = self.patch_embed(x)                     # (B, hidden_dim, gh, gw)
        B, D, gh, gw = h.shape
        return h.permute(0, 2, 3, 1).reshape(B, gh * gw, D)

    def _unpatchify(self, h: torch.Tensor) -> torch.Tensor:
        """(B, N, out_channels*P*P) → (B, out_channels, H, W)"""
        B = h.shape[0]
        gh = gw = int(self.num_patches ** 0.5)
        P = self.cfg.patch_size
        C_out = self.cfg.out_channels
        h = h.reshape(B, gh, gw, C_out, P, P)
        h = h.permute(0, 3, 1, 4, 2, 5).reshape(B, C_out, gh * P, gw * P)
        return h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
            t: (B,) or (B, 1) time in [0, 1]
            y: (B,) class labels, or None
            cond: (B, cond_channels, H, W) condition image, or None
        Returns:
            (B, out_channels, H, W)  — vector field estimate
        """
        # Build conditioning embedding: time + optional class
        emb = self.time_mlp(t)                              # (B, hidden_dim)
        if self.label_emb is not None and y is not None:
            y_emb = self.label_proj(self.label_emb(y))     # (B, hidden_dim)
            emb = emb + y_emb

        # Channel-concatenate condition image before patch embedding
        if cond is not None:
            x = torch.cat([x, cond], dim=1)  # (B, in_channels + cond_channels, H, W)

        # Tokenise image
        tokens = self._patchify(x) + self.pos_emb          # (B, N, hidden_dim)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, emb)

        # Decode back to image space
        tokens = self.out_norm(tokens)                      # (B, N, hidden_dim)
        tokens = self.out_proj(tokens)                      # (B, N, C_out * P * P)
        return self._unpatchify(tokens)                     # (B, out_channels, H, W)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@register_model("vit2d")
def build_vit2d(config_dict: dict) -> ConditionalVectorField:
    """Build a ViT2D from a raw config dict (called by the model registry)."""
    cfg = ViT2DConfig(**config_dict)
    return ViT2D(cfg)
