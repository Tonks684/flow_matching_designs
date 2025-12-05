# scripts/train_mnist.py

import argparse
import yaml
import torch

from flow_matching_designs.data.mnist import get_mnist_dataloaders
from flow_matching_designs.sampling.sampler import MNISTSampler
from flow_matching_designs.math.schedules import LinearAlpha, LinearBeta
from flow_matching_designs.math.paths import GaussianConditionalProbabilityPath
from flow_matching_designs.training.trainer import CFGTrainer
from flow_matching_designs.training.callbacks import CheckpointCallback, CSVLoggerCallback
from flow_matching_designs.models.registry import build_model
from flow_matching_designs.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/mnist_baseline.yaml")
    p.add_argument("--model", type=str, default=None,
                   help="Model architecture name (overrides config.model_name), e.g. 'unet2d'")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["training"]["seed"])

    # --------- Data ---------
    train_loader, _ = get_mnist_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    p_data = MNISTSampler(data_root=cfg["data"]["data_dir"])
    p_simple_shape = [
        cfg["model"]["in_channels"],
        cfg["model"]["image_size"],
        cfg["model"]["image_size"],
    ]

    # --------- Path / schedules ---------
    alpha = LinearAlpha()
    beta = LinearBeta()
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        p_simple_shape=p_simple_shape,
        alpha=alpha,
        beta=beta,
    )

    # --------- Model (via registry) ---------
    model_name = args.model or cfg.get("model_name", "unet2d")
    model = build_model(model_name, cfg["model"]).to(device)

    # --------- Trainer ---------
    trainer = CFGTrainer(
        path=path,
        model=model,
        eta=cfg["training"]["eta"],
        null_label=cfg["training"]["null_label"],
    )

    trainer.train(
        num_epochs=cfg["training"]["num_epochs"],
        device=device,
        train_loader=train_loader,
        lr=cfg["training"]["lr"],
        callbacks=[
            CheckpointCallback(save_every=5),
            CSVLoggerCallback(),
        ],
    )


if __name__ == "__main__":
    main()
