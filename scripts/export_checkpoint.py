import argparse
import os

import torch
import yaml

from flow_matching_designs.models.registry import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/mnist_baseline.yaml")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to training checkpoint (.pt) with state_dict",
    )
    p.add_argument("--outdir", type=str, default="exports")
    p.add_argument(
        "--basename",
        type=str,
        default="mnist_unet",
        help="Base name for exported files (no extension)",
    )
    p.add_argument(
        "--script",
        action="store_true",
        help="Also export a TorchScript traced model",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture name (overrides config.model_name), e.g. 'unet2d'",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.outdir, exist_ok=True)

    # ----- Build model (on CPU) -----
    model_name = args.model or cfg.get("model_name", "unet2d")
    model = build_model(model_name, cfg["model"])
    model.eval()

    # ----- Load checkpoint -----
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ----- Export plain CPU state_dict -----
    export_path = os.path.join(args.outdir, f"{args.basename}.pt")
    torch.save(model.state_dict(), export_path)
    print(f"Exported CPU model state_dict to {export_path}")

    # ----- Optional: TorchScript export -----
    if args.script:
        C = cfg["model"]["in_channels"]
        H = cfg["model"]["image_size"]
        W = cfg["model"]["image_size"]

        example_x = torch.randn(1, C, H, W)
        example_t = torch.tensor([0.5])               # (1,)
        example_y = torch.tensor([0], dtype=torch.long)

        scripted = torch.jit.trace(model, (example_x, example_t, example_y))
        script_path = os.path.join(args.outdir, f"{args.basename}_scripted.pt")
        scripted.save(script_path)
        print(f"Exported TorchScript model to {script_path}")


if __name__ == "__main__":
    main()
