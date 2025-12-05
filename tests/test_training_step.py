import torch
from torch.utils.data import DataLoader, TensorDataset

from flow_matching_designs.math.paths import GaussianConditionalProbabilityPath
from flow_matching_designs.math.schedules import LinearAlpha, LinearBeta
from flow_matching_designs.models.registry import build_model
from flow_matching_designs.training.losses import cfg_flow_matching_loss
from flow_matching_designs.training.trainer import CFGTrainer
from flow_matching_designs.sampling.sampler import Sampleable
from torch import nn


class DummyDataDist(nn.Module, Sampleable):
    """
    Very small dummy data distribution used only to satisfy
    GaussianConditionalProbabilityPath's p_data argument.
    """

    def __init__(self, shape=(1, 8, 8), num_classes: int = 10):
        super().__init__()
        self.shape = shape
        self.num_classes = num_classes
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples: int):
        x = torch.randn(num_samples, *self.shape, device=self.dummy.device)
        y = torch.randint(0, self.num_classes, (num_samples,), device=self.dummy.device)
        return x, y


def _small_model_and_path():
    # Small model config for faster tests
    model_cfg = {
        "in_channels": 1,
        "out_channels": 1,
        "num_classes": 10,
        "image_size": 8,
        "base_channels": 8,
        "channel_multipliers": [1, 2],
        "num_res_blocks": 1,
        "time_embedding_dim": 32,
    }
    model = build_model("unet2d", model_cfg)

    p_data = DummyDataDist(shape=(1, 8, 8))
    alpha = LinearAlpha()
    beta = LinearBeta()
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        p_simple_shape=[1, 8, 8],
        alpha=alpha,
        beta=beta,
    )
    return model, path


def test_cfg_flow_matching_loss_runs_and_backprops():
    torch.manual_seed(0)
    device = torch.device("cpu")

    model, path = _small_model_and_path()
    model.to(device)

    B, C, H, W = 4, 1, 8, 8
    x_data = torch.randn(B, C, H, W, device=device)
    y = torch.randint(0, 10, (B,), device=device)

    loss = cfg_flow_matching_loss(
        model=model,
        path=path,
        x_data=x_data,
        y=y,
        eta=0.1,
        device=device,
        null_label=10,  # matches num_classes in UNet2DConfig
    )

    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_cfg_trainer_one_epoch_on_tiny_dataset():
    torch.manual_seed(0)
    device = torch.device("cpu")

    model, path = _small_model_and_path()
    trainer = CFGTrainer(path=path, model=model, eta=0.1, null_label=10)

    # Tiny synthetic dataset (no MNIST dependency)
    B, C, H, W = 8, 1, 8, 8
    x = torch.randn(B, C, H, W)
    y = torch.randint(0, 10, (B,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    # Just check that this runs without error
    trainer.train(
        num_epochs=1,
        device=device,
        train_loader=loader,
        lr=1e-3,
        callbacks=[],  # keep it simple
    )
