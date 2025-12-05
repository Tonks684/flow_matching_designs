from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn

from flow_matching_designs.math.schedules import Schedule  # LinearAlpha/LinearBeta implement this
from flow_matching_designs.sampling.sampler import Sampleable


# ---------------------------------------------------------------------
# Isotropic Gaussian prior p_simple
# ---------------------------------------------------------------------
class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn.

    Samples x ~ N(0, std^2 I) with shape (num_samples, *shape).
    """

    def __init__(self, shape: List[int], std: float = 1.0):
        """
        Args:
            shape: shape of each sample, e.g. [C, H, W]
            std: standard deviation of the Gaussian
        """
        super().__init__()
        self.shape = list(shape)
        self.std = float(std)
        # This buffer ensures we always know which device to sample on.
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, None]:
        x = self.std * torch.randn(num_samples, *self.shape, device=self.dummy.device)
        return x, None


# ---------------------------------------------------------------------
# Abstract conditional probability path
# ---------------------------------------------------------------------
class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths.
    """

    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple  # e.g. IsotropicGaussian
        self.p_data = p_data      # e.g. dataset distribution wrapper

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = ∫ p_t(x|z) p(z) dz.

        Args:
            t: time tensor of shape (num_samples,) or (num_samples, 1)
        Returns:
            x: samples from p_t(x), shape (num_samples, C, H, W)
        """
        if t.dim() == 1:
            t = t[:, None]  # (num_samples, 1)
        num_samples = t.shape[0]

        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples)  # (num_samples, C, H, W)

        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t)  # (num_samples, C, H, W)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y.

        Args:
            num_samples: number of samples
        Returns:
            z: (num_samples, C, H, W)
            y: (num_samples, label_dim) or (num_samples,) depending on implementation
        """
        raise NotImplementedError

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z).

        Args:
            z: conditioning variable, shape (num_samples, C, H, W)
            t: time, shape (num_samples,) or (num_samples, 1)
        Returns:
            x: samples from p_t(x|z), shape (num_samples, C, H, W)
        """
        raise NotImplementedError

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z).

        Args:
            x: position variable, shape (num_samples, C, H, W)
            z: conditioning variable, shape (num_samples, C, H, W)
            t: time, shape (num_samples,) or (num_samples, 1)
        Returns:
            conditional_vector_field: shape (num_samples, C, H, W)
        """
        raise NotImplementedError

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score ∇_x log p_t(x|z).

        Args:
            x: position variable, shape (num_samples, C, H, W)
            z: conditioning variable, shape (num_samples, C, H, W)
            t: time, shape (num_samples,) or (num_samples, 1)
        Returns:
            conditional_score: shape (num_samples, C, H, W)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------
# Gaussian conditional probability path
# ---------------------------------------------------------------------
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    A Gaussian conditional path with schedules alpha(t), beta(t).

    Typical form:
        x_t = alpha(t) * z + beta(t) * epsilon, epsilon ~ N(0, I)
    """

    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Schedule, beta: Schedule):
        p_simple = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        super().__init__(p_simple=p_simple, p_data=p_data)
        self.alpha = alpha
        self.beta = beta

    def _broadcast_time(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure t has shape (B, 1) for schedule evaluation.
        """
        if t.dim() == 1:
            t = t[:, None]
        return t

    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y.

        Delegates to self.p_data.sample(num_samples).

        Returns:
            z: (num_samples, C, H, W)
            y: labels or None, depending on p_data implementation
        """
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z).

        Args:
            z: conditioning variable, shape (num_samples, C, H, W)
            t: time, shape (num_samples,) or (num_samples, 1)
        Returns:
            x: samples from p_t(x|z), shape (num_samples, C, H, W)
        """
        t = self._broadcast_time(t, z)               # (B,1)
        alpha_t = self.alpha(t)                      # (B,1)
        beta_t = self.beta(t)                        # (B,1)
        eps = torch.randn_like(z)

        # broadcast (B,1) -> (B,1,1,1)
        alpha_b = alpha_t[:, :, None, None]
        beta_b = beta_t[:, :, None, None]

        x = alpha_b * z + beta_b * eps
        return x

    def conditional_vector_field(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z) for the Gaussian path.

        Uses:
            u_t(x|z) = (α'(t) - β'(t)/β(t) * α(t)) * z + (β'(t)/β(t)) * x

        Args:
            x: (B, C, H, W) position variable
            z: (B, C, H, W) conditioning variable
            t: (B,) or (B, 1) time

        Returns:
            u_t(x|z): (B, C, H, W)
        """
        # Ensure t has shape (B,1)
        t = self._broadcast_time(t, x)  # (B,1)

        alpha_t = self.alpha(t)          # (B,1)
        beta_t = self.beta(t)           # (B,1)
        dt_alpha_t = self.alpha.dt(t)    # (B,1)
        dt_beta_t = self.beta.dt(t)      # (B,1)

        # Avoid numerical issues when beta_t is tiny
        eps = 1e-5
        beta_safe = beta_t.clamp(min=eps)

        # Coefficients in (B,1)
        coeff_z = dt_alpha_t - (dt_beta_t / beta_safe) * alpha_t
        coeff_x = dt_beta_t / beta_safe

        # Broadcast to (B,1,1,1)
        coeff_z = coeff_z[:, :, None, None]
        coeff_x = coeff_x[:, :, None, None]

        return coeff_z * z + coeff_x * x

    def conditional_score(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates the conditional score ∇_x log p_t(x|z) for the Gaussian path.

        For:
            x_t = α(t) z + β(t) ε,   ε ~ N(0, I)

        the score is:
            ∇_x log p_t(x|z) = (α(t) z - x) / β(t)^2

        Args:
            x: (B, C, H, W)
            z: (B, C, H, W)
            t: (B,) or (B, 1)

        Returns:
            score: (B, C, H, W)
        """
        t = self._broadcast_time(t, x)       # (B,1)

        alpha_t = self.alpha(t)              # (B,1)
        beta_t = self.beta(t)                # (B,1)

        eps = 1e-5
        beta_safe = beta_t.clamp(min=eps)

        # (B,1,1,1)
        alpha_b = alpha_t[:, :, None, None]
        beta_b = beta_safe[:, :, None, None]

        return (alpha_b * z - x) / (beta_b ** 2)
