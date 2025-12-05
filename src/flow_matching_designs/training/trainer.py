from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Iterable

import torch
from torch import nn
from tqdm.auto import tqdm

from flow_matching_designs.math.paths import GaussianConditionalProbabilityPath
from flow_matching_designs.models.conditional_vector_field import ConditionalVectorField
from flow_matching_designs.training.losses import cfg_flow_matching_loss


# 1 MiB is 1,048,576 bytes based on powers of 2 (binary),
# which matches how model parameters are actually stored.
MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, batch, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        num_epochs: int,
        device: torch.device,
        train_loader: Iterable,
        lr: float = 1e-3,
        callbacks: Optional[Iterable] = None,
    ) -> None:
        """
        Generic training loop that consumes a PyTorch DataLoader.

        Args:
            num_epochs: number of epochs
            device: device to train on
            train_loader: DataLoader yielding (x, y) batches
            lr: learning rate
        """
        callbacks = callbacks or []
        self.device = device
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        opt = self.get_optimizer(lr)

        for cb in callbacks:
            cb.on_train_start(self)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            num_steps = 0

            for cb in callbacks:
                cb.on_epoch_start(self, epoch)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                opt.zero_grad(set_to_none=True)
                loss = self.get_train_loss(batch=batch, device=device)
                loss.backward()
                opt.step()

                running_loss += loss.item()
                num_steps += 1
                avg_loss = running_loss / num_steps
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                for cb in callbacks:
                    cb.on_batch_end(self, epoch, batch_idx, loss.item())

            epoch_loss = running_loss / max(num_steps, 1)
            print(f"[Epoch {epoch}] loss: {epoch_loss:.4f}")

            for cb in callbacks:
                cb.on_epoch_end(self, epoch, epoch_loss)

        self.model.eval()
        for cb in callbacks:
            cb.on_train_end(self)

        print("Training finished.")


class CFGTrainer(Trainer):
    """
    Trainer for classifier-free guidance flow matching using batches from a DataLoader.
    """

    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        model: ConditionalVectorField,
        eta: float,
        null_label: Optional[int] = None,
    ):
        assert 0.0 < eta < 1.0, "eta must be in (0, 1)"
        super().__init__(model)
        self.path = path
        self.eta = eta

        # If dataset knows num_classes, infer default null label from it
        if null_label is None:
            self.null_label = getattr(path.p_data, "num_classes", 10)
        else:
            self.null_label = null_label

    def get_train_loss(self, batch, device: torch.device) -> torch.Tensor:
        # Expect batch to be (images, labels) from DataLoader
        x_data, y = batch
        return cfg_flow_matching_loss(
            model=self.model,
            path=self.path,
            x_data=x_data,
            y=y,
            eta=self.eta,
            device=device,
            null_label=self.null_label,
        )
