import argparse
import os

import torch
import yaml
from torchvision.utils import save_image

from flow_matching_designs.models.registry import build_model
from flow_matching_designs.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/mnist_baseline.yaml")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument(
        "--class_idx",
        type=int,
        default=-1,
        help="Digit class to condition on (0-9). -1 = unconditional.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of Euler steps from t=0 -> t=1",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale; 0 = no guidance.",
    )
    p.add_argument("--outdir", type=str, default="samples")
    p.add_argument("--outfile", type=str, default="mnist_samples.png")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture name (overrides config.model_name), e.g. 'unet2d'",
    )
    return p.parse_args()


def euler_integrate_flow(
    model: torch.nn.Module,
    x0: torch.Tensor,
    y: torch.Tensor | None,
    null_label: int,
    steps: int,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Simple Euler integrator for dx/dt = u_theta(x, t, y).

    If guidance_scale > 0 and y is not None, uses classifier-free guidance:
        u_cfg = (1 + s) * u(x, t, y) - s * u(x, t, y_null)
    """
    model.eval()
    x = x0.to(device)
    B = x.size(0)

    t_values = torch.linspace(0.0, 1.0, steps, device=device)
    dt = 1.0 / max(steps - 1, 1)

    if y is not None:
        y = y.to(device)

    with torch.no_grad():
        for t in t_values:
            t_batch = torch.full((B,), t, device=device)

            if guidance_scale > 0.0 and y is not None:
                # conditional
                u_cond = model(x, t_batch, y)
                # unconditional (null label)
                y_null = torch.full_like(y, null_label)
                u_uncond = model(x, t_batch, y_null)
                u = (1.0 + guidance_scale) * u_cond - guidance_scale * u_uncond
            else:
                # no CFG or no labels: unconditional only
                u = model(x, t_batch, y if y is not None else None)

            x = x + dt * u

    return x


def main():
    args = parse_args()

    # ----- Load config -----
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["training"]["seed"])

    os.makedirs(args.outdir, exist_ok=True)

    # ----- Build model from registry -----
    model_name = args.model or cfg.get("model_name", "unet2d")
    model = build_model(model_name, cfg["model"]).to(device)

    # ----- Load checkpoint -----
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ----- Prepare initial noise x0 from simple Gaussian prior -----
    B = args.num_samples
    C = cfg["model"]["in_channels"]
    H = cfg["model"]["image_size"]
    W = cfg["model"]["image_size"]

    x0 = torch.randn(B, C, H, W, device=device)

    # ----- Labels for sampling -----
    null_label = cfg["training"]["null_label"]

    if args.class_idx >= 0:
        # condition on a specific digit
        y = torch.full((B,), args.class_idx, dtype=torch.long, device=device)
    else:
        # unconditional: either pure flow or CFG with null only
        y = None

    # ----- Integrate the flow -----
    xT = euler_integrate_flow(
        model=model,
        x0=x0,
        y=y,
        null_label=null_label,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=device,
    )

    # ----- Convert from [-1,1] -> [0,1] for saving (given Normalize((0.5,), (0.5,))) -----
    samples = (xT.clamp(-1, 1) + 1.0) / 2.0

    outfile = os.path.join(args.outdir, args.outfile)
    save_image(samples, outfile, nrow=int(B ** 0.5))
    print(f"Saved samples to {outfile}")


if __name__ == "__main__":
    main()
