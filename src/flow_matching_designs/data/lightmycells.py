"""
PyTorch Dataset and DataLoader for the Light My Cells dataset.

Dataset reference:
    "2D Multimodal Image Collection for Fluorescence Prediction from
    Transmitted Light Microscopy", Kauffmann et al., Scientific Data 2026.
    BioImage Archive accession: S-BIAD1047

Expected directory layout after download and organisation:

    root/
    ├── train/
    │   ├── nucleus/
    │   │   ├── image_001_IM.tif   ← transmitted light (BF / PC / DIC)
    │   │   ├── image_001_GT.tif   ← fluorescence ground truth
    │   │   └── ...
    │   ├── mitochondria/
    │   ├── tubulin/
    │   └── actin/
    └── holdout/
        └── <same structure>

Each item yielded by the DataLoader is a 3-tuple:
    (target, cond, label)
    target : (1, H, W) float32 in [-1, 1]  — fluorescence image to generate
    cond   : (1, H, W) float32 in [-1, 1]  — transmitted-light condition image
    label  : int                            — organelle class index

Organelle label map:
    nucleus=0, mitochondria=1, tubulin=2, actin=3

This is compatible with CFGTrainer which unpacks 3-element batches automatically.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


ORGANELLE_LABELS: Dict[str, int] = {
    "nucleus": 0,
    "mitochondria": 1,
    "tubulin": 2,
    "actin": 3,
}


def _load_tiff_as_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    """
    Load a single-channel OME-TIFF, normalise to [-1, 1], resize, and
    return as a (1, H, W) float32 tensor.
    """
    img = tifffile.imread(str(path))         # (H, W) or (1, H, W) or (H, W, 1)

    # Squeeze to 2D
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got shape {img.shape} for {path}")

    # Convert to float32 and normalise to [0, 1]
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)

    # (H, W) → (1, H, W) tensor, then resize to image_size
    tensor = torch.from_numpy(img).unsqueeze(0)          # (1, H, W)
    tensor = TF.resize(tensor, [image_size, image_size],
                       interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

    # [0, 1] → [-1, 1]
    tensor = tensor * 2.0 - 1.0
    return tensor


class LightMyCellsDataset(Dataset):
    """
    Paired transmitted-light → fluorescence dataset from Light My Cells.

    Args:
        root:         Path to the split directory, e.g. ``data/train`` or ``data/holdout``.
        organelles:   Which organelle subfolders to include.  Defaults to all four.
        image_size:   Spatial size each image is resized to (square).
        transform:    Optional additional transform applied to both target and cond
                      tensors as a ``(target, cond)`` tuple.
    """

    def __init__(
        self,
        root: str | Path,
        organelles: Optional[List[str]] = None,
        image_size: int = 512,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform
        self.organelles = organelles or list(ORGANELLE_LABELS.keys())

        self.samples: List[Tuple[Path, Path, int]] = []  # (im_path, gt_path, label)
        self._index_files()

    def _index_files(self) -> None:
        for organelle in self.organelles:
            label = ORGANELLE_LABELS[organelle]
            organ_dir = self.root / organelle
            if not organ_dir.is_dir():
                continue

            im_files = sorted(organ_dir.glob("*_IM.tif"))
            for im_path in im_files:
                gt_path = Path(str(im_path).replace("_IM.tif", "_GT.tif"))
                if gt_path.exists():
                    self.samples.append((im_path, gt_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No paired *_IM.tif / *_GT.tif files found under {self.root}. "
                "Check that the dataset has been downloaded and organised correctly."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        im_path, gt_path, label = self.samples[idx]

        cond   = _load_tiff_as_tensor(im_path, self.image_size)   # transmitted light
        target = _load_tiff_as_tensor(gt_path, self.image_size)   # fluorescence

        if self.transform is not None:
            target, cond = self.transform(target, cond)

        return target, cond, label


def get_lightmycells_dataloaders(
    data_dir: str | Path,
    organelles: Optional[List[str]] = None,
    image_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and holdout DataLoaders for the Light My Cells dataset.

    Each batch is a 3-tuple ``(target, cond, label)`` compatible with
    ``CFGTrainer`` which automatically unpacks 3-element batches.

    Args:
        data_dir:    Root data directory containing ``train/`` and ``holdout/``
                     subdirectories.
        organelles:  Organelle subsets to include; defaults to all four.
        image_size:  Resize images to this square resolution.
        batch_size:  DataLoader batch size.
        num_workers: DataLoader worker processes.
        pin_memory:  Pin memory for faster GPU transfer.

    Returns:
        (train_loader, holdout_loader)
    """
    data_dir = Path(data_dir)

    train_ds   = LightMyCellsDataset(data_dir / "train",   organelles, image_size)
    holdout_ds = LightMyCellsDataset(data_dir / "holdout", organelles, image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    holdout_loader = DataLoader(
        holdout_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, holdout_loader
