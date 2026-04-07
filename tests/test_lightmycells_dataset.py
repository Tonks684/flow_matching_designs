"""
Tests for the Light My Cells dataset and dataloader.

All tests use the synthetic ``lightmycells_dir`` fixture from conftest.py —
no real dataset download is required.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from flow_matching_designs.data.lightmycells import (
    LightMyCellsDataset,
    ORGANELLE_LABELS,
    get_lightmycells_dataloaders,
)


IMAGE_SIZE = 16  # small for speed in tests


# ---------------------------------------------------------------------------
# Dataset indexing
# ---------------------------------------------------------------------------

def test_dataset_finds_all_pairs(lightmycells_dir):
    """Train split: 2 nucleus + 1 mitochondria + 1 tubulin + 1 actin = 5 pairs."""
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    assert len(ds) == 5


def test_dataset_holdout_split(lightmycells_dir):
    """Holdout split contains 1 nucleus pair."""
    ds = LightMyCellsDataset(lightmycells_dir / "holdout", image_size=IMAGE_SIZE)
    assert len(ds) == 1


def test_dataset_organelle_filter(lightmycells_dir):
    """Filtering to a single organelle returns only that organelle's pairs."""
    ds = LightMyCellsDataset(
        lightmycells_dir / "train",
        organelles=["nucleus"],
        image_size=IMAGE_SIZE,
    )
    assert len(ds) == 2
    # All labels should be nucleus=0
    labels = {ds[i][2] for i in range(len(ds))}
    assert labels == {ORGANELLE_LABELS["nucleus"]}


def test_dataset_raises_when_no_files(tmp_path):
    """FileNotFoundError when the directory exists but has no TIFF pairs."""
    empty = tmp_path / "train" / "nucleus"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        LightMyCellsDataset(tmp_path / "train", image_size=IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Item shapes and types
# ---------------------------------------------------------------------------

def test_dataset_item_returns_three_elements(lightmycells_dir):
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    item = ds[0]
    assert len(item) == 3, "Expected (target, cond, label)"


def test_dataset_target_shape(lightmycells_dir):
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    target, cond, label = ds[0]
    assert target.shape == (1, IMAGE_SIZE, IMAGE_SIZE), f"Bad target shape: {target.shape}"


def test_dataset_cond_shape(lightmycells_dir):
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    target, cond, label = ds[0]
    assert cond.shape == (1, IMAGE_SIZE, IMAGE_SIZE), f"Bad cond shape: {cond.shape}"


def test_dataset_label_is_int(lightmycells_dir):
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    _, _, label = ds[0]
    assert isinstance(label, int)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_dataset_target_range(lightmycells_dir):
    """Target pixel values should lie in [-1, 1]."""
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    for i in range(len(ds)):
        target, _, _ = ds[i]
        assert target.min() >= -1.0 - 1e-5
        assert target.max() <= 1.0 + 1e-5


def test_dataset_cond_range(lightmycells_dir):
    """Condition pixel values should lie in [-1, 1]."""
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    for i in range(len(ds)):
        _, cond, _ = ds[i]
        assert cond.min() >= -1.0 - 1e-5
        assert cond.max() <= 1.0 + 1e-5


def test_dataset_tensors_are_float32(lightmycells_dir):
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    target, cond, _ = ds[0]
    assert target.dtype == torch.float32
    assert cond.dtype == torch.float32


# ---------------------------------------------------------------------------
# Organelle label mapping
# ---------------------------------------------------------------------------

def test_dataset_label_values_are_valid(lightmycells_dir):
    """Every label must be one of the four defined organelle indices."""
    valid_labels = set(ORGANELLE_LABELS.values())
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    for i in range(len(ds)):
        _, _, label = ds[i]
        assert label in valid_labels, f"Unexpected label {label}"


def test_organelle_label_map_completeness():
    """All four organelles must be in the label map."""
    assert set(ORGANELLE_LABELS.keys()) == {"nucleus", "mitochondria", "tubulin", "actin"}
    assert set(ORGANELLE_LABELS.values()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------

def test_dataloader_batch_shapes(lightmycells_dir):
    """DataLoader should yield (target, cond, label) with correct batch shapes."""
    ds = LightMyCellsDataset(lightmycells_dir / "train", image_size=IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    assert len(batch) == 3
    target, cond, label = batch
    assert target.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert cond.shape   == (2, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert label.shape  == (2,)


def test_get_lightmycells_dataloaders(lightmycells_dir):
    """Factory function should return two DataLoaders."""
    train_loader, holdout_loader = get_lightmycells_dataloaders(
        data_dir=lightmycells_dir,
        image_size=IMAGE_SIZE,
        batch_size=2,
        num_workers=0,
    )
    assert isinstance(train_loader,   DataLoader)
    assert isinstance(holdout_loader, DataLoader)

    # Train loader has 5 items → ceil(5/2) = 3 batches
    assert len(train_loader.dataset) == 5
    # Holdout loader has 1 item
    assert len(holdout_loader.dataset) == 1
