import torch

from flow_matching_designs.models.registry import build_model
from flow_matching_designs.models.conditional_vector_field import (
    ConditionalVectorField,
)


def _small_unet_cfg():
    # Use a smaller UNet than the training config so tests are faster
    return {
        "name": "unet2d",
        "in_channels": 1,
        "out_channels": 1,
        "num_classes": 10,
        "image_size": 32,
        "base_channels": 16,
        "channel_multipliers": [1, 2],
        "num_res_blocks": 1,
        "time_embedding_dim": 32,
    }


def test_unet2d_is_registered():
    cfg = _small_unet_cfg()
    name = cfg.pop("name")
    model = build_model(name, cfg)

    assert isinstance(model, ConditionalVectorField)
    assert hasattr(model, "forward")


def test_unet2d_forward_shapes_and_grad():
    cfg = _small_unet_cfg()
    name = cfg.pop("name")
    model = build_model(name, cfg)

    B, C, H, W = 4, cfg["in_channels"], cfg["image_size"], cfg["image_size"]
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)  # (B,)
    y = torch.randint(0, cfg["num_classes"], (B,))

    out = model(x, t, y)
    assert out.shape == (B, cfg["out_channels"], H, W)

    loss = out.mean()
    loss.backward()

    # At least one parameter should have non-zero gradient
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_unet2d_works_without_labels():
    cfg = _small_unet_cfg()
    name = cfg.pop("name")
    model = build_model(name, cfg)

    B, C, H, W = 2, cfg["in_channels"], cfg["image_size"], cfg["image_size"]
    x = torch.randn(B, C, H, W)
    t = torch.rand(B, 1)  # (B,1)

    out = model(x, t)  # y=None
    assert out.shape == (B, cfg["out_channels"], H, W)
