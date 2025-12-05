import torch
from torch import nn

from flow_matching_designs.math.paths import (
    IsotropicGaussian,
    GaussianConditionalProbabilityPath,
)
from flow_matching_designs.math.schedules import LinearAlpha, LinearBeta
from flow_matching_designs.sampling.sampler import Sampleable


class DummyDataDist(nn.Module, Sampleable):
    """
    Simple sampleable distribution used only for testing paths.
    Samples standard normal images and dummy integer labels.
    """

    def __init__(self, shape=(1, 32, 32), num_classes: int = 10):
        super().__init__()
        self.shape = shape
        self.num_classes = num_classes
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples: int):
        x = torch.randn(num_samples, *self.shape, device=self.dummy.device)
        y = torch.randint(0, self.num_classes, (num_samples,), device=self.dummy.device)
        return x, y


def test_isotropic_gaussian_sample_shape_and_device():
    shape = [3, 8, 8]
    dist = IsotropicGaussian(shape=shape, std=0.5)
    dist = dist.to("cpu")

    x, labels = dist.sample(4)
    assert labels is None
    assert x.shape == (4, *shape)
    assert x.device == dist.dummy.device
    # Check approximate std (very weak check, just to see it's non-degenerate)
    assert x.std() > 0.1


def test_gaussian_conditional_path_shapes():
    B = 5
    shape = (1, 16, 16)
    p_data = DummyDataDist(shape=shape)
    alpha = LinearAlpha()
    beta = LinearBeta()

    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        p_simple_shape=list(shape),
        alpha=alpha,
        beta=beta,
    )

    # Use a concrete conditioning variable z
    z, _ = p_data.sample(B)
    t = torch.rand(B)  # (B,)

    x_t = path.sample_conditional_path(z, t)
    assert x_t.shape == z.shape

    vf = path.conditional_vector_field(x_t, z, t)
    assert vf.shape == z.shape
    assert torch.isfinite(vf).all()

    score = path.conditional_score(x_t, z, t)
    assert score.shape == z.shape
    assert torch.isfinite(score).all()


def test_sample_marginal_path_uses_internal_p_data():
    B = 3
    shape = (1, 8, 8)
    p_data = DummyDataDist(shape=shape)
    alpha = LinearAlpha()
    beta = LinearBeta()
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        p_simple_shape=list(shape),
        alpha=alpha,
        beta=beta,
    )

    t = torch.rand(B)  # (B,)
    x = path.sample_marginal_path(t)
    assert x.shape == (B, *shape)
