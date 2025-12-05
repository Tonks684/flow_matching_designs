from __future__ import annotations

from typing import Callable, Dict, Any

from .conditional_vector_field import ConditionalVectorField

MODEL_REGISTRY: Dict[str, Callable[[dict], ConditionalVectorField]] = {}


def register_model(name: str):
    """
    Decorator to register a model builder under a given name.

    Usage:

        @register_model("unet2d")
        def build_unet2d(cfg: dict) -> ConditionalVectorField:
            ...
    """
    def decorator(fn: Callable[[dict], ConditionalVectorField]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def build_model(name: str, cfg: dict) -> ConditionalVectorField:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](cfg)


# Import model modules here so that their registration decorators run
# (side-effect imports to populate MODEL_REGISTRY).
from . import unet  # noqa: F401  # ensure UNet is registered
