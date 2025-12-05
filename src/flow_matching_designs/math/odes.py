from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from flow_matching_designs.models.conditional_vector_field import ConditionalVectorField


# ----------------------------------------------------------------------
# ODE base class
# ----------------------------------------------------------------------
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient f(xt, t) of the ODE d x_t / dt = f(x_t, t).

        Args:
            xt: state at time t, shape (B, C, H, W)
            t:  time, shape (B,) or (B, 1)

        Returns:
            drift: shape (B, C, H, W)
        """
        raise NotImplementedError


class CFGVectorFieldODE(ODE):
    """
    ODE defined by a learned conditional vector field u_θ(x, t, y),
    with optional classifier-free guidance.

    The underlying model must implement ConditionalVectorField.
    """

    def __init__(
        self,
        net: ConditionalVectorField,
        null_label: int,
        guidance_scale: float = 0.0,
    ):
        """
        Args:
            net:          ConditionalVectorField model, u_θ(x, t, y)
            null_label:   label index used for unconditional (null) conditioning
            guidance_scale: guidance strength s. If > 0, uses:

                u_cfg = (1 + s) * u(x, t, y) - s * u(x, t, y_null)

                where y_null is filled with null_label.

        """
        super().__init__()
        self.net = net
        self.null_label = null_label
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            t: (B,) or (B, 1)
            y: (B,) or None. If None, uses unconditional (null label) only.

        Returns:
            drift: (B, C, H, W)
        """
        # No labels or guidance_scale == 0 → just use unconditional with null label
        if y is None or self.guidance_scale <= 0.0:
            if y is None:
                # all null labels
                B = x.size(0)
                y_null = torch.full((B,), self.null_label, dtype=torch.long, device=x.device)
            else:
                y_null = torch.full_like(y, self.null_label)

            return self.net(x, t, y_null)

        # With CFG: combine conditional & unconditional fields
        y = y.to(x.device)
        t = t.to(x.device)

        u_cond = self.net(x, t, y)

        y_null = torch.full_like(y, self.null_label)
        u_uncond = self.net(x, t, y_null)

        s = self.guidance_scale
        u_cfg = (1.0 + s) * u_cond - s * u_uncond
        return u_cfg


# ----------------------------------------------------------------------
# SDE base class
# ----------------------------------------------------------------------
class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient f(xt, t) of the SDE d x_t = f(x_t, t) dt + g(x_t, t) dW_t.

        Args:
            xt: state at time t, shape (B, C, H, W)
            t:  time, shape (B,) or (B, 1)

        Returns:
            drift: shape (B, C, H, W)
        """
        raise NotImplementedError

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient g(xt, t) of the SDE.

        Args:
            xt: state at time t, shape (B, C, H, W)
            t:  time, shape (B,) or (B, 1)

        Returns:
            diffusion: shape (B, C, H, W)
        """
        raise NotImplementedError
