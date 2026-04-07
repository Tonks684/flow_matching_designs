"""
Generates synthetic placeholder images for the README.

Produces representative examples of:
  - MNIST generated digits (class-conditional)
  - Light My Cells brightfield condition and fluorescence target

Run from the repo root:
    python docs/generate_readme_images.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(arr: np.ndarray, name: str) -> None:
    """Save a uint8 numpy array (H, W) or (H, W, 3) as a PNG."""
    img = Image.fromarray(arr)
    img.save(OUT / name)
    print(f"  wrote {OUT / name}")


def _normalise_u8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# MNIST-style digits (simple bitmapped strokes)
# ---------------------------------------------------------------------------

def _digit_image(digit: int, size: int = 64) -> np.ndarray:
    """Draw a rough grey-on-black bitmapped digit."""
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 3

    strokes = {
        0: lambda: draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=220, width=4),
        1: lambda: (draw.line([cx, cy - r, cx, cy + r], fill=220, width=5),
                    draw.line([cx - 6, cy - r + 6, cx, cy - r], fill=220, width=5)),
        7: lambda: (draw.line([cx - r, cy - r, cx + r, cy - r], fill=220, width=5),
                    draw.line([cx + r, cy - r, cx - 4, cy + r], fill=220, width=5)),
    }
    strokes.get(digit, strokes[0])()

    arr = np.array(img, dtype=np.float32)
    # Add mild Gaussian noise
    arr += RNG.normal(0, 8, arr.shape).astype(np.float32)
    arr = arr.clip(0, 255)
    blurred = np.array(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)))
    return blurred


def make_mnist_grid(digits: list[int], cols: int = 4, cell: int = 64,
                    pad: int = 4) -> np.ndarray:
    rows = (len(digits) + cols - 1) // cols
    H = rows * (cell + pad) + pad
    W = cols * (cell + pad) + pad
    canvas = np.zeros((H, W), dtype=np.uint8)
    for i, d in enumerate(digits):
        r, c = divmod(i, cols)
        y = pad + r * (cell + pad)
        x = pad + c * (cell + pad)
        canvas[y:y + cell, x:x + cell] = _digit_image(d, cell)
    return canvas


# ---------------------------------------------------------------------------
# Synthetic brightfield (transmitted light) image
# Looks like faint, grey, out-of-focus cells with halos.
# ---------------------------------------------------------------------------

def make_brightfield(size: int = 256) -> np.ndarray:
    canvas = np.ones((size, size), dtype=np.float32) * 0.75

    n_cells = 12
    for _ in range(n_cells):
        cx = RNG.integers(30, size - 30)
        cy = RNG.integers(30, size - 30)
        rx = RNG.integers(12, 28)
        ry = RNG.integers(10, 22)
        angle = RNG.uniform(0, np.pi)

        yy, xx = np.ogrid[:size, :size]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = cos_a * (xx - cx) + sin_a * (yy - cy)
        yr = -sin_a * (xx - cx) + cos_a * (yy - cy)
        mask = (xr / rx) ** 2 + (yr / ry) ** 2

        # Cell body — slightly darker than background
        canvas[mask < 1.0] *= 0.78
        # Bright halo ring
        halo = (mask >= 0.85) & (mask < 1.15)
        canvas[halo] = canvas[halo] * 1.18

    # Mild Gaussian noise + blur
    canvas += RNG.normal(0, 0.015, canvas.shape).astype(np.float32)
    canvas = canvas.clip(0, 1)
    blurred = np.array(
        Image.fromarray(_normalise_u8(canvas)).filter(ImageFilter.GaussianBlur(1.5))
    )
    return blurred


# ---------------------------------------------------------------------------
# Synthetic fluorescence image — dark bg, bright puncta / ring structures
# ---------------------------------------------------------------------------

def make_fluorescence(size: int = 256, organelle: str = "nucleus") -> np.ndarray:
    canvas = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]

    if organelle == "nucleus":
        # Bright filled ellipses (nuclei)
        n = 10
        for _ in range(n):
            cx, cy = RNG.integers(25, size - 25, size=2)
            rx, ry = RNG.integers(10, 22), RNG.integers(10, 18)
            intensity = RNG.uniform(0.6, 1.0)
            d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            canvas[d < 1.0] = np.maximum(canvas[d < 1.0],
                                          intensity * (1 - d[d < 1.0]) ** 0.5)

    elif organelle == "mitochondria":
        # Thin elongated bright filaments
        n = 30
        for _ in range(n):
            cx, cy = RNG.integers(10, size - 10, size=2)
            length = RNG.integers(8, 25)
            angle = RNG.uniform(0, np.pi)
            intensity = RNG.uniform(0.5, 0.95)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = cos_a * (xx - cx) + sin_a * (yy - cy)
            yr = -sin_a * (xx - cx) + cos_a * (yy - cy)
            mask = (np.abs(xr) < length) & (np.abs(yr) < 2.5)
            canvas[mask] = np.maximum(canvas[mask], intensity)

    else:
        # Generic diffuse puncta
        n = 80
        for _ in range(n):
            cx, cy = RNG.integers(5, size - 5, size=2)
            r = RNG.uniform(1.5, 4.0)
            intensity = RNG.uniform(0.4, 1.0)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            contrib = intensity * np.exp(-(d ** 2) / (2 * r ** 2))
            canvas = np.maximum(canvas, contrib)

    canvas += RNG.normal(0, 0.02, canvas.shape).astype(np.float32)
    canvas = canvas.clip(0, 1)
    blurred = np.array(
        Image.fromarray(_normalise_u8(canvas)).filter(ImageFilter.GaussianBlur(0.8))
    )
    return blurred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating README demo images...")

    # MNIST grids
    for digit, name in [(0, "mnist_class0.png"), (1, "mnist_class1.png"), (7, "mnist_class7.png")]:
        grid = make_mnist_grid([digit] * 8, cols=4, cell=56, pad=3)
        _save(grid, name)

    # Light My Cells demo — brightfield condition
    bf = make_brightfield(256)
    _save(bf, "lmc_bf.png")

    # Fluorescence targets
    fluor = make_fluorescence(256, "nucleus")
    _save(fluor, "lmc_fluor.png")

    # "Predicted" — slightly blurred version of ground truth to simulate model output
    pred = np.array(Image.fromarray(fluor).filter(ImageFilter.GaussianBlur(1.2)))
    pred = (pred.astype(np.float32) * 0.88 +
            RNG.normal(0, 4, pred.shape).clip(-10, 10)).clip(0, 255).astype(np.uint8)
    _save(pred, "lmc_pred.png")

    print("Done.")
