from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn
from torchvision import datasets, transforms


class Sampleable(ABC):
    """
    Distribution which can be sampled from.
    """

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            num_samples: the desired number of samples

        Returns:
            samples: shape (num_samples, ...)
            labels:  shape (num_samples, ...) or None
        """
        raise NotImplementedError


class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset.

    Provides:
        - .sample(num_samples) -> (images, labels)
        - .num_classes = 10 (for convenience)
    """

    def __init__(self, data_root: str = "./data"):
        super().__init__()

        self.dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
            ]),
        )

        # This buffer is only used to track device; it will move with .to(...)
        self.register_buffer("dummy", torch.zeros(1))

        # Expose num_classes so other components (e.g. trainer/path) can infer defaults
        self.num_classes = 10

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            num_samples: the desired number of samples

        Returns:
            samples: shape (num_samples, C, H, W)
            labels:  shape (num_samples,)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples_list, labels_list = zip(*[self.dataset[i] for i in indices])

        samples = torch.stack(samples_list).to(self.dummy.device)
        labels = torch.tensor(labels_list, dtype=torch.int64, device=self.dummy.device)
        return samples, labels
