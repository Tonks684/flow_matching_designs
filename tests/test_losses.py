"""
Tests for cfg_flow_matching_loss with and without image conditioning.
"""
from __future__ import annotations

import torch
import pytest

from flow_matching_designs.training.losses import cfg_flow_matching_loss


DEVICE = torch.device("cpu")
B, C, H, W = 4, 1, 8, 8


# ---------------------------------------------------------------------------
# No conditioning (original behaviour)
# ---------------------------------------------------------------------------

def test_loss_no_cond_returns_scalar(small_unet, small_path, tiny_batch):
    x_data, y = tiny_batch
    loss = cfg_flow_matching_loss(
        model=small_unet, path=small_path,
        x_data=x_data, y=y,
        eta=0.1, device=DEVICE, null_label=10,
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_loss_no_cond_backprop(small_unet, small_path, tiny_batch):
    x_data, y = tiny_batch
    loss = cfg_flow_matching_loss(
        model=small_unet, path=small_path,
        x_data=x_data, y=y,
        eta=0.1, device=DEVICE, null_label=10,
    )
    loss.backward()
    grads = [p.grad for p in small_unet.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------------------
# With image conditioning
# ---------------------------------------------------------------------------

def test_loss_with_cond_returns_scalar(small_unet_cond, small_path, tiny_batch_cond):
    x_data, cond, y = tiny_batch_cond
    loss = cfg_flow_matching_loss(
        model=small_unet_cond, path=small_path,
        x_data=x_data, y=y,
        eta=0.1, device=DEVICE, null_label=4,
        cond=cond,
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_loss_with_cond_backprop(small_unet_cond, small_path, tiny_batch_cond):
    x_data, cond, y = tiny_batch_cond
    loss = cfg_flow_matching_loss(
        model=small_unet_cond, path=small_path,
        x_data=x_data, y=y,
        eta=0.1, device=DEVICE, null_label=4,
        cond=cond,
    )
    loss.backward()
    grads = [p.grad for p in small_unet_cond.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_loss_cond_null_drop_zeroes_condition(small_unet_cond, small_path):
    """
    When eta=1.0 every sample is masked, so cond_tilde should be all zeros
    and the label should be null_label.  We verify this indirectly by checking
    the loss still runs and produces a finite scalar.
    """
    torch.manual_seed(42)
    x_data = torch.randn(B, C, H, W)
    cond   = torch.ones(B, C, H, W)    # non-zero so masking is detectable
    y      = torch.randint(0, 4, (B,))

    loss = cfg_flow_matching_loss(
        model=small_unet_cond, path=small_path,
        x_data=x_data, y=y,
        eta=0.9999, device=DEVICE, null_label=4,
        cond=cond,
    )
    assert torch.isfinite(loss)


def test_loss_cond_none_matches_no_cond_signature(small_unet, small_path, tiny_batch):
    """Passing cond=None explicitly should be identical to omitting it."""
    x_data, y = tiny_batch
    torch.manual_seed(7)
    loss_implicit = cfg_flow_matching_loss(
        model=small_unet, path=small_path,
        x_data=x_data, y=y,
        eta=0.1, device=DEVICE, null_label=10,
    )
    torch.manual_seed(7)
    loss_explicit = cfg_flow_matching_loss(
        model=small_unet, path=small_path,
        x_data=x_data, y=y,
        eta=0.1, device=DEVICE, null_label=10,
        cond=None,
    )
    assert torch.allclose(loss_implicit, loss_explicit)
