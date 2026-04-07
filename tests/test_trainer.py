"""
Tests for CFGTrainer with two-element (x, y) and three-element (x, cond, y) batches.
"""
from __future__ import annotations

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from flow_matching_designs.training.trainer import CFGTrainer


DEVICE = torch.device("cpu")
B, C, H, W = 8, 1, 8, 8


def _make_loader(with_cond: bool, num_classes: int = 10) -> DataLoader:
    x = torch.randn(B, C, H, W)
    y = torch.randint(0, num_classes, (B,))
    if with_cond:
        cond = torch.randn(B, C, H, W)
        ds   = TensorDataset(x, cond, y)
    else:
        ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4, shuffle=False)


# ---------------------------------------------------------------------------
# Two-element batches (class-conditional, no image condition)
# ---------------------------------------------------------------------------

def test_trainer_two_element_batch_runs(small_unet, small_path):
    trainer = CFGTrainer(path=small_path, model=small_unet, eta=0.1, null_label=10)
    loader  = _make_loader(with_cond=False)
    trainer.train(num_epochs=1, device=DEVICE, train_loader=loader, lr=1e-3)


def test_trainer_two_element_loss_decreases_over_epochs(small_unet, small_path):
    """Loss should not be nan/inf across multiple epochs."""
    torch.manual_seed(0)
    trainer = CFGTrainer(path=small_path, model=small_unet, eta=0.1, null_label=10)
    loader  = _make_loader(with_cond=False)
    # Just check it runs 3 epochs without error
    trainer.train(num_epochs=3, device=DEVICE, train_loader=loader, lr=1e-3)


# ---------------------------------------------------------------------------
# Three-element batches (image-to-image conditioning)
# ---------------------------------------------------------------------------

def test_trainer_three_element_batch_runs(small_unet_cond, small_path):
    trainer = CFGTrainer(path=small_path, model=small_unet_cond, eta=0.1, null_label=4)
    loader  = _make_loader(with_cond=True, num_classes=4)
    trainer.train(num_epochs=1, device=DEVICE, train_loader=loader, lr=1e-3)


def test_trainer_three_element_batch_backprops(small_unet_cond, small_path):
    """Parameters should receive gradients when training with an image condition."""
    trainer = CFGTrainer(path=small_path, model=small_unet_cond, eta=0.1, null_label=4)
    loader  = _make_loader(with_cond=True, num_classes=4)

    # Zero out grads first
    for p in small_unet_cond.parameters():
        p.grad = None

    trainer.train(num_epochs=1, device=DEVICE, train_loader=loader, lr=1e-3)

    grads = [p.grad for p in small_unet_cond.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


# ---------------------------------------------------------------------------
# get_train_loss API
# ---------------------------------------------------------------------------

def test_get_train_loss_two_element(small_unet, small_path):
    trainer = CFGTrainer(path=small_path, model=small_unet, eta=0.1, null_label=10)
    x = torch.randn(4, C, H, W)
    y = torch.randint(0, 10, (4,))
    loss = trainer.get_train_loss(batch=(x, y), device=DEVICE)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_get_train_loss_three_element(small_unet_cond, small_path):
    trainer = CFGTrainer(path=small_path, model=small_unet_cond, eta=0.1, null_label=4)
    x    = torch.randn(4, C, H, W)
    cond = torch.randn(4, C, H, W)
    y    = torch.randint(0, 4, (4,))
    loss = trainer.get_train_loss(batch=(x, cond, y), device=DEVICE)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
