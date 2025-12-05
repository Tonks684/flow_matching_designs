# src/flow_matching_designs/math/schedules.py

from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class Schedule(ABC):
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the schedule at times t.
        t: (B,) or (B,1)
        Returns: (B,1)
        """
        raise NotImplementedError


# -------------------------------------------------------------------
# Alpha(t): alpha(0)=0, alpha(1)=1
# -------------------------------------------------------------------

class LinearAlpha(Schedule):
    """
    Implements alpha(t) = t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # make sure t is (B,1)
        if t.dim() == 1:
            t = t[:, None]
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return torch.ones_like(t)


# -------------------------------------------------------------------
# Beta(t): beta(0)=1, beta(1)=0
# -------------------------------------------------------------------

class LinearBeta(Schedule):
    """
    Implements beta(t) = 1 - t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return 1.0 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return -torch.ones_like(t)
