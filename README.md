# 🚀 Flow Matching Designs

## A modular, extensible library for generative modelling using Flow Matching

### flow_matching_designs is a research-ready and production-oriented framework for training conditional flow-matching models on 2D image datasets.
The repo is designed to scale from MNIST to arbitrary conditional image generators, with:

- a model registry (registry.py) so new architectures plug in easily
- a config-driven pipeline
- modular math components (probability paths, ODEs, schedules)
- reusable training loop with callbacks
- clean separation of concerns in the repo layout

This project currently includes implementations of:
- **Conditional UNet 2D** — convolutional encoder-decoder with residual blocks
- **ViT2D** — Vision Transformer with AdaLN-Zero conditioning (DiT-style)

Both architectures are trained with classifier-free guidance flow matching on MNIST.

## Repo Structure
```
flow_matching_designs/
│
├── configs/                     # YAML config files for training
│   ├── mnist_baseline.yaml
│   ├── mnist_cfg_strong.yaml
│   └── mnist_vit.yaml
│
├── scripts/                     # Run scripts (train/sample/export)
│   ├── train_mnist.py
│   ├── sample_mnist.py
│   └── export_checkpoint.py
│
├── src/flow_matching_designs/
│   ├── data/                    # Dataset loaders, wrappers
│   ├── math/                    # Flow-matching math components
│   │   ├── schedules.py
│   │   ├── paths.py
│   │   ├── odes.py
│   │   └── simulators.py
│   │
│   ├── models/                  # Architectures + model registry
│   │   ├── unet.py
│   │   ├── unet_blocks.py
│   │   ├── vit.py
│   │   ├── conditional_vector_field.py
│   │   └── registry.py
│   │
│   ├── sampling/                # Dataset samplers
│   │   └── sampler.py
│   │
│   ├── training/                # Losses, trainer, callbacks
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── callbacks.py
│   │
│   └── utils/                   # Seed, logging, viz, distributed utils
│       ├── seed.py
│       ├── distributed.py
│       └── viz.py
│
├── notebooks/                   # Jupyter experiments
├── tests/                       # Unit tests
├── README.md
├── requirements.txt
├── pyproject.toml               # Build/installation metadata
└── Dockerfile
```

### Installation (dev mode):
```
git clone https://github.com/Tonks684/flow_matching_designs.git
cd flow_matching_designs
pip install -e .
```

Train MNIST with UNet (baseline):
```
PYTHONPATH=./src python scripts/train_mnist.py --config configs/mnist_baseline.yaml
```

Train MNIST with ViT:
```
PYTHONPATH=./src python scripts/train_mnist.py --config configs/mnist_vit.yaml
```
## 🖼 Sampling Images

Generate MNIST digits from a trained model:
```
python scripts/sample_mnist.py \
    --checkpoint ckpts/mnist_unet2d.pt \
    --n_samples 64 \
    --outfile samples.png
```
## 📤 Exporting a Checkpoint
```
python scripts/export_checkpoint.py \
    --checkpoint ckpts/mnist_unet2d.pt \
    --out models/exported_model.pt
```

## 🤖 Available Models

### UNet2D (`unet2d`)
Convolutional encoder-decoder with residual blocks, downsampling/upsampling, and skip connections. Time and class conditioning are injected via sinusoidal embeddings added to each residual block.

Config key: `model_name: "unet2d"`

### ViT2D (`vit2d`)
Vision Transformer with **AdaLN-Zero** conditioning (following [DiT](https://arxiv.org/abs/2212.09748)):

1. **Patchify** — input image split into `patch_size × patch_size` patches, each linearly embedded to `hidden_dim` tokens
2. **Positional embedding** — learnable embedding added to each token
3. **Transformer blocks** — multi-head self-attention + MLP, with time/class conditioning modulating LayerNorm scale, shift, and residual gates
4. **Unpatchify** — tokens projected back to pixel space and reshaped to `(B, C, H, W)`

Config key: `model_name: "vit2d"`

Key hyperparameters (see `configs/mnist_vit.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 4 | Patch size in pixels (image_size must be divisible) |
| `hidden_dim` | 384 | Token/embedding dimension |
| `num_heads` | 6 | Attention heads |
| `num_layers` | 6 | Transformer depth |
| `mlp_ratio` | 4.0 | MLP hidden dim multiplier |

---

## 🧩 Adding New Models

All models register themselves via:

```
from .registry import register_model

@register_model("my_new_arch")
def build_model(cfg_dict):
    return MyModelClass(**cfg_dict)

```
Then in a new config file
```
model:
  name: my_new_arch
  hidden_dim: 512
  depth: 6
```
Training script will automatically pick this up
```
model = build_model(cfg["model"]["name"], cfg["model"])
```

## 📦 Docker Usage

Build container:
```
docker build -t flow-matching .
```

Run training inside container:
```
docker run --gpus all -v $(pwd):/workspace flow-matching \
    python scripts/train_mnist.py --config configs/mnist_baseline.yaml
```

