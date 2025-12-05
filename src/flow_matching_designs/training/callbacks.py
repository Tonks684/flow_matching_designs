"""
Object to hook into events during training
"""

from __future__ import annotations
import os
import csv
from typing import Optional
import torch


# ------------------------------------------------------------
# Base callback interface
# ------------------------------------------------------------
class Callback:
    def on_train_start(self, trainer): pass
    def on_epoch_start(self, trainer, epoch: int): pass
    def on_batch_end(self, trainer, epoch: int, batch_idx: int, loss: float): pass
    def on_epoch_end(self, trainer, epoch: int, avg_loss: float): pass
    def on_train_end(self, trainer): pass


# ------------------------------------------------------------
# Checkpoint callback
# ------------------------------------------------------------
class CheckpointCallback(Callback):
    """
    Saves model weights every `save_every` epochs.
    """
    def __init__(self, save_dir: str = "checkpoints", save_every: int = 1):
        self.save_dir = save_dir
        self.save_every = save_every
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, epoch: int, avg_loss: float):
        if (epoch + 1) % self.save_every == 0:
            path = os.path.join(self.save_dir, f"epoch_{epoch+1}.pt")
            torch.save(trainer.model.state_dict(), path)
            print(f"[Callback] Saved checkpoint to {path}")


# ------------------------------------------------------------
# CSV Logger callback
# ------------------------------------------------------------
class CSVLoggerCallback(Callback):
    """
    Logs losses to a CSV file: epoch, batch_loss, avg_loss.
    """
    def __init__(self, filepath: str = "training_log.csv"):
        self.filepath = filepath
        self.file = None
        self.writer = None

    def on_train_start(self, trainer):
        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["epoch", "batch_idx", "batch_loss", "avg_loss"])

    def on_batch_end(self, trainer, epoch: int, batch_idx: int, batch_loss: float):
        self.writer.writerow([epoch, batch_idx, batch_loss, None])

    def on_epoch_end(self, trainer, epoch: int, avg_loss: float):
        self.writer.writerow([epoch, None, None, avg_loss])

    def on_train_end(self, trainer):
        if self.file:
            self.file.close()
            print(f"[Callback] Training log saved to {self.filepath}")


# ------------------------------------------------------------
# Optional: sampling callback
# ------------------------------------------------------------
class SampleCallback(Callback):
    """
    Generates sample images every N epochs during training.

    Requires:
        - a sampler function, e.g. sample_fn(model, device)
        - output_dir for saving generated images
    """

    def __init__(self, sample_fn, output_dir="samples", every=5):
        self.sample_fn = sample_fn
        self.output_dir = output_dir
        self.every = every
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, trainer, epoch: int, avg_loss: float):
        if (epoch + 1) % self.every == 0:
            img = self.sample_fn(trainer.model, trainer.device)  # you define this externally
            path = os.path.join(self.output_dir, f"sample_epoch_{epoch+1}.png")
            img.save(path)
            print(f"[Callback] Saved sample image to {path}")
