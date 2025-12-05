from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class ConditionalVectorField(nn.Module, ABC):
    """
    Abstract base class for conditional vector fields u_θ(x, t, y).

    Implementations should take:
        x: (B, C, H, W)
        t: (B,) or (B, 1)
        y: (B,) or None
    and return:
        u_θ(x, t, y): (B, C, H, W)
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            t: (B,) or (B, 1) time in [0,1]
            y: (B,) class labels, or None
        Returns:
            u_θ(x, t, y): (B, C, H, W)
        """
        raise NotImplementedError
