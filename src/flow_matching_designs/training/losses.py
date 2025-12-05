from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from flow_matching_designs.math.paths import GaussianConditionalProbabilityPath


def cfg_flow_matching_loss(
    model: nn.Module,
    path: GaussianConditionalProbabilityPath,
    x_data: torch.Tensor,          # batch of images from DataLoader
    y: torch.Tensor,               # batch of labels from DataLoader
    eta: float,
    device: torch.device,
    null_label: int,
) -> torch.Tensor:
    """
    Classifier-free guidance flow-matching loss using a batch from a DataLoader.

    Args:
        model: conditional vector field u_theta(x, t, y_tilde)
        path: probability path, must implement sample_conditional_path and conditional_vector_field
        x_data: (B, C, H, W) data batch (acts as z in the notation)
        y: (B,) class labels
        eta: probability of dropping the label (CFG mask rate)
        device: torch.device
        null_label: index used as the "null" class
    """
    # z is just the data batch
    z = x_data.to(device)          # (B, C, H, W)
    y = y.to(device)               # (B,)

    batch_size = z.size(0)

    # 1) Classifier-free masking
    mask = (torch.rand(batch_size, device=device) < eta)
    y_tilde = y.clone()
    y_tilde[mask] = null_label

    # 2) Sample t and x_t
    t = torch.rand(batch_size, device=device)       # (B,)
    x = path.sample_conditional_path(z, t)          # (B, C, H, W)
    x = x.to(device)

    # 3) Target and predicted vector fields
    with torch.no_grad():
        target_vf = path.conditional_vector_field(x, z, t)  # (B, C, H, W)

    pred_vf = model(x, t, y_tilde)                          # (B, C, H, W)

    # --- SAFETY CHECKS (so we don't get a cryptic TypeError) ---
    if target_vf is None:
        raise RuntimeError(
            "path.conditional_vector_field(x, z, t) returned None. "
            "Check GaussianConditionalProbabilityPath.conditional_vector_field implementation."
        )

    if pred_vf is None:
        raise RuntimeError(
            "model(x, t, y_tilde) returned None. "
            "Check ConditionalUNet2D.forward (it must end with `return h`)."
        )

    if not isinstance(target_vf, torch.Tensor):
        raise TypeError(f"Expected target_vf to be Tensor, got {type(target_vf)}")

    if not isinstance(pred_vf, torch.Tensor):
        raise TypeError(f"Expected pred_vf to be Tensor, got {type(pred_vf)}")

    # 4) MSE loss
    loss = torch.mean((pred_vf - target_vf) ** 2)
    return loss
