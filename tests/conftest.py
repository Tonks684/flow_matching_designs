"""
Shared pytest fixtures for the flow_matching_designs test suite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from torch.utils.data import DataLoader, TensorDataset

from flow_matching_designs.math.paths import GaussianConditionalProbabilityPath
from flow_matching_designs.math.schedules import LinearAlpha, LinearBeta
from flow_matching_designs.models.registry import build_model
from flow_matching_designs.sampling.sampler import Sampleable
from torch import nn


# ---------------------------------------------------------------------------
# Synthetic Light My Cells directory fixture
# ---------------------------------------------------------------------------

ORGANELLES = ["nucleus", "mitochondria", "tubulin", "actin"]
_IMG_SIZE = 32  # small so tests are fast


def _write_tiff(path: Path, h: int = _IMG_SIZE, w: int = _IMG_SIZE) -> None:
    """Write a random uint16 synthetic TIFF to *path*."""
    arr = (np.random.randint(0, 65535, (h, w), dtype=np.uint16))
    tifffile.imwrite(str(path), arr)


@pytest.fixture()
def lightmycells_dir(tmp_path: Path) -> Path:
    """
    Minimal synthetic Light My Cells directory:

        tmp/
        ├── train/
        │   ├── nucleus/      image_001_IM.tif  image_001_GT.tif
        │   │                 image_002_IM.tif  image_002_GT.tif
        │   ├── mitochondria/ image_001_IM.tif  image_001_GT.tif
        │   ├── tubulin/      image_001_IM.tif  image_001_GT.tif
        │   └── actin/        image_001_IM.tif  image_001_GT.tif
        └── holdout/
            └── nucleus/      image_001_IM.tif  image_001_GT.tif
    """
    rng = np.random.default_rng(0)

    for split, counts in [("train", {"nucleus": 2, "mitochondria": 1, "tubulin": 1, "actin": 1}),
                           ("holdout", {"nucleus": 1})]:
        for organelle, n in counts.items():
            organ_dir = tmp_path / split / organelle
            organ_dir.mkdir(parents=True)
            for i in range(1, n + 1):
                _write_tiff(organ_dir / f"image_{i:03d}_IM.tif")
                _write_tiff(organ_dir / f"image_{i:03d}_GT.tif")

    return tmp_path


# ---------------------------------------------------------------------------
# Small model / path helpers shared across test modules
# ---------------------------------------------------------------------------

class _DummyDataDist(nn.Module, Sampleable):
    def __init__(self, shape=(1, 8, 8), num_classes: int = 10):
        super().__init__()
        self.shape = shape
        self.num_classes = num_classes
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples: int):
        x = torch.randn(num_samples, *self.shape, device=self.dummy.device)
        y = torch.randint(0, self.num_classes, (num_samples,), device=self.dummy.device)
        return x, y


@pytest.fixture()
def small_unet():
    cfg = dict(in_channels=1, out_channels=1, num_classes=10, image_size=8,
               base_channels=8, channel_multipliers=[1, 2], num_res_blocks=1,
               time_embedding_dim=32)
    return build_model("unet2d", cfg)


@pytest.fixture()
def small_unet_cond():
    cfg = dict(in_channels=1, out_channels=1, num_classes=4, image_size=8,
               base_channels=8, channel_multipliers=[1, 2], num_res_blocks=1,
               time_embedding_dim=32, cond_channels=1)
    return build_model("unet2d", cfg)


@pytest.fixture()
def small_vit():
    cfg = dict(in_channels=1, out_channels=1, num_classes=10, image_size=8,
               patch_size=2, hidden_dim=32, num_heads=2, num_layers=2,
               mlp_ratio=2.0, time_embedding_dim=32)
    return build_model("vit2d", cfg)


@pytest.fixture()
def small_vit_cond():
    cfg = dict(in_channels=1, out_channels=1, num_classes=4, image_size=8,
               patch_size=2, hidden_dim=32, num_heads=2, num_layers=2,
               mlp_ratio=2.0, time_embedding_dim=32, cond_channels=1)
    return build_model("vit2d", cfg)


@pytest.fixture()
def small_path():
    p_data = _DummyDataDist(shape=(1, 8, 8))
    return GaussianConditionalProbabilityPath(
        p_data=p_data,
        p_simple_shape=[1, 8, 8],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    )


@pytest.fixture()
def tiny_batch():
    """Returns (x_data, y) tensors of shape (4, 1, 8, 8) and (4,)."""
    x = torch.randn(4, 1, 8, 8)
    y = torch.randint(0, 10, (4,))
    return x, y


@pytest.fixture()
def tiny_batch_cond():
    """Returns (x_data, cond, y) tensors for image-to-image training."""
    x    = torch.randn(4, 1, 8, 8)
    cond = torch.randn(4, 1, 8, 8)
    y    = torch.randint(0, 4, (4,))
    return x, cond, y
