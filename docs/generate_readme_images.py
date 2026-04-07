"""
Generates synthetic dataset-preview images for the README.

Shows what the raw data looks like — no trained model outputs.

  mnist_examples.png         — grid of MNIST-style digit examples
  lmc_pairs.png              — paired brightfield | fluorescence for all
                               four Light My Cells organelle channels

Run from the repo root:
    python docs/generate_readme_images.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
CELL = 64   # pixels per image cell in grids
PAD  = 4    # padding between cells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(img: Image.Image, name: str) -> None:
    img.save(OUT / name)
    print(f"  wrote {OUT / name}")


def _normalise_u8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """(H, W) uint8 → (H, W, 3) uint8."""
    return np.stack([arr, arr, arr], axis=-1)


def _label_strip(text: str, width: int, height: int = 18,
                 bg: tuple = (30, 30, 30), fg: tuple = (200, 200, 200)) -> np.ndarray:
    """Create a small labelled strip below an image."""
    strip = Image.new("RGB", (width, height), color=bg)
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, 2), text, fill=fg, font=font)
    return np.array(strip)


# ---------------------------------------------------------------------------
# MNIST-style digit examples
# ---------------------------------------------------------------------------

def _digit_cell(digit: int, size: int = CELL) -> np.ndarray:
    img  = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 3

    def _0(): draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=210, width=4)
    def _1():
        draw.line([cx, cy - r, cx, cy + r], fill=210, width=5)
        draw.line([cx - 7, cy - r + 7, cx, cy - r], fill=210, width=5)
    def _2():
        draw.arc([cx - r, cy - r, cx + r, cy], 200, 360, fill=210, width=4)
        draw.line([cx + r, cy, cx - r + 4, cy + r], fill=210, width=4)
        draw.line([cx - r + 4, cy + r, cx + r, cy + r], fill=210, width=4)
    def _3():
        draw.arc([cx - r, cy - r, cx + r, cy], 315, 135, fill=210, width=4)
        draw.arc([cx - r, cy, cx + r, cy + r], 315, 135, fill=210, width=4)
    def _4():
        draw.line([cx + r // 2, cy - r, cx - r // 2, cy], fill=210, width=4)
        draw.line([cx - r // 2, cy, cx + r, cy], fill=210, width=4)
        draw.line([cx + r // 2, cy - r, cx + r // 2, cy + r], fill=210, width=4)
    def _5():
        draw.line([cx + r // 2, cy - r, cx - r, cy - r], fill=210, width=4)
        draw.line([cx - r, cy - r, cx - r, cy], fill=210, width=4)
        draw.arc([cx - r, cy, cx + r, cy + r], 270, 180, fill=210, width=4)
        draw.line([cx - r, cy, cx, cy], fill=210, width=4)
    def _6():
        draw.arc([cx - r, cy - r, cx + r, cy + r], 90, 0, fill=210, width=4)
        draw.ellipse([cx - r + 6, cy - 4, cx + r - 6, cy + r - 2], outline=210, width=4)
    def _7():
        draw.line([cx - r, cy - r, cx + r, cy - r], fill=210, width=4)
        draw.line([cx + r, cy - r, cx - 4, cy + r], fill=210, width=4)
    def _8():
        draw.ellipse([cx - r + 4, cy - r, cx + r - 4, cy], outline=210, width=4)
        draw.ellipse([cx - r, cy, cx + r, cy + r], outline=210, width=4)
    def _9():
        draw.ellipse([cx - r + 4, cy - r, cx + r - 4, cy + 2], outline=210, width=4)
        draw.line([cx + r - 4, cy, cx + r - 4, cy + r], fill=210, width=4)

    [_0, _1, _2, _3, _4, _5, _6, _7, _8, _9][digit]()

    arr = np.array(img, dtype=np.float32)
    arr += RNG.normal(0, 6, arr.shape).astype(np.float32)
    arr = arr.clip(0, 255)
    return np.array(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)))


def make_mnist_examples(digits_per_row: int = 5, size: int = CELL) -> Image.Image:
    """Grid showing one example of each digit 0–9."""
    digits = list(range(10))
    cols = digits_per_row
    rows = (len(digits) + cols - 1) // cols
    strip_h = 18

    W = cols * (size + PAD) + PAD
    H = rows * (size + strip_h + PAD) + PAD
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    for i, d in enumerate(digits):
        r, c = divmod(i, cols)
        x = PAD + c * (size + PAD)
        y = PAD + r * (size + strip_h + PAD)
        cell = _to_rgb(_digit_cell(d, size))
        canvas[y:y + size, x:x + size] = cell
        label = _label_strip(f"digit {d}", size, strip_h)
        canvas[y + size:y + size + strip_h, x:x + size] = label

    return Image.fromarray(canvas)


# ---------------------------------------------------------------------------
# Synthetic transmitted-light (brightfield) image
# ---------------------------------------------------------------------------

def make_brightfield(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.ones((size, size), dtype=np.float32) * 0.72

    for _ in range(10):
        cx = rng.integers(20, size - 20)
        cy = rng.integers(20, size - 20)
        rx = rng.integers(10, 22)
        ry = rng.integers(8, 18)
        angle = rng.uniform(0, np.pi)
        yy, xx = np.ogrid[:size, :size]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = cos_a * (xx - cx) + sin_a * (yy - cy)
        yr = -sin_a * (xx - cx) + cos_a * (yy - cy)
        mask = (xr / rx) ** 2 + (yr / ry) ** 2
        canvas[mask < 1.0] *= 0.80
        canvas[(mask >= 0.85) & (mask < 1.15)] *= 1.15

    canvas += rng.normal(0, 0.012, canvas.shape).astype(np.float32)
    canvas = canvas.clip(0, 1)
    return _normalise_u8(
        np.array(Image.fromarray(_normalise_u8(canvas)).filter(ImageFilter.GaussianBlur(1.2)))
        .astype(np.float32) / 255
    )


# ---------------------------------------------------------------------------
# Synthetic fluorescence images per organelle
# ---------------------------------------------------------------------------

def make_fluorescence(organelle: str, size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]

    if organelle == "nucleus":
        for _ in range(8):
            cx, cy = rng.integers(18, size - 18, size=2)
            rx, ry = rng.integers(9, 18), rng.integers(8, 14)
            intensity = rng.uniform(0.65, 1.0)
            d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            canvas[d < 1.0] = np.maximum(canvas[d < 1.0],
                                          intensity * (1 - d[d < 1.0]) ** 0.4)

    elif organelle == "mitochondria":
        for _ in range(25):
            cx, cy = rng.integers(8, size - 8, size=2)
            length = rng.integers(6, 20)
            angle  = rng.uniform(0, np.pi)
            intensity = rng.uniform(0.5, 0.9)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = cos_a * (xx - cx) + sin_a * (yy - cy)
            yr = -sin_a * (xx - cx) + cos_a * (yy - cy)
            mask = (np.abs(xr) < length) & (np.abs(yr) < 2.5)
            canvas[mask] = np.maximum(canvas[mask], intensity)

    elif organelle == "tubulin":
        # Long thin microtubule-like fibres
        for _ in range(12):
            cx, cy = rng.integers(5, size - 5, size=2)
            length = rng.integers(20, 55)
            angle  = rng.uniform(0, np.pi)
            intensity = rng.uniform(0.4, 0.85)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = cos_a * (xx - cx) + sin_a * (yy - cy)
            yr = -sin_a * (xx - cx) + cos_a * (yy - cy)
            mask = (np.abs(xr) < length) & (np.abs(yr) < 1.8)
            canvas[mask] = np.maximum(canvas[mask], intensity)

    elif organelle == "actin":
        # Diffuse cortical / filamentous actin
        for _ in range(60):
            cx, cy = rng.integers(4, size - 4, size=2)
            r = rng.uniform(1.2, 3.5)
            intensity = rng.uniform(0.3, 0.85)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            canvas = np.maximum(canvas, intensity * np.exp(-(d ** 2) / (2 * r ** 2)))

    canvas += rng.normal(0, 0.018, canvas.shape).astype(np.float32)
    canvas = canvas.clip(0, 1)
    return _normalise_u8(
        np.array(Image.fromarray(_normalise_u8(canvas)).filter(ImageFilter.GaussianBlur(0.7)))
        .astype(np.float32) / 255
    )


# ---------------------------------------------------------------------------
# Light My Cells: paired grid (BF | Fluorescence) × 4 organelles
# ---------------------------------------------------------------------------

ORGANELLES = ["nucleus", "mitochondria", "tubulin", "actin"]
FLUOR_COLOURS = {
    "nucleus":      (100, 180, 255),   # blue
    "mitochondria": (255, 120,  80),   # orange-red
    "tubulin":      ( 80, 230, 130),   # green
    "actin":        (230, 200,  60),   # yellow
}


def _colourise(grey: np.ndarray, colour: tuple) -> np.ndarray:
    """Apply a false colour to a greyscale image. grey: (H,W) uint8 → (H,W,3) uint8."""
    out = np.zeros((*grey.shape, 3), dtype=np.uint8)
    for i, c in enumerate(colour):
        out[:, :, i] = (grey.astype(np.float32) * c / 255).clip(0, 255).astype(np.uint8)
    return out


def make_lmc_pairs(cell: int = 128) -> Image.Image:
    """
    2-column grid: Transmitted Light (grey) | Fluorescence (false colour)
    One row per organelle.
    """
    strip_h = 20
    col_w   = cell
    cols    = 2
    rows    = len(ORGANELLES)

    W = cols * col_w + (cols + 1) * PAD
    H = strip_h + PAD + rows * (cell + strip_h + PAD) + PAD  # header + data rows + bottom pad
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (15, 15, 15)

    col_labels = ["Transmitted Light", "Fluorescence"]
    for ci, lbl in enumerate(col_labels):
        x = PAD + ci * (col_w + PAD)
        hdr = _label_strip(lbl, col_w, strip_h, bg=(50, 50, 50), fg=(230, 230, 230))
        canvas[PAD:PAD + strip_h, x:x + col_w] = hdr

    for ri, organelle in enumerate(ORGANELLES):
        y = PAD + strip_h + PAD + ri * (cell + strip_h + PAD)

        # Brightfield (grey → RGB)
        bf   = make_brightfield(cell, seed=ri)
        bf_rgb = _to_rgb(bf)
        x_bf = PAD
        canvas[y:y + cell, x_bf:x_bf + col_w] = bf_rgb

        # Fluorescence (false coloured)
        fl   = make_fluorescence(organelle, cell, seed=ri * 10)
        fl_c = _colourise(fl, FLUOR_COLOURS[organelle])
        x_fl = PAD + col_w + PAD
        canvas[y:y + cell, x_fl:x_fl + col_w] = fl_c

        # Organelle label under the fluorescence image
        lbl_strip = _label_strip(organelle, col_w, strip_h)
        canvas[y + cell:y + cell + strip_h, x_fl:x_fl + col_w] = lbl_strip

    return Image.fromarray(canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating README dataset-preview images...")

    _save(make_mnist_examples(), "mnist_examples.png")
    _save(make_lmc_pairs(),      "lmc_pairs.png")

    print("Done.")
