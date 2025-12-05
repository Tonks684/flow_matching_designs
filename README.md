# 🚀 Flow Matching Designs

## A modular, extensible library for generative modelling using Flow Matching

### flow_matching_designs is a research-ready and production-oriented framework for training conditional flow-matching models on 2D image datasets.
The repo is designed to scale from MNIST to arbitrary conditional image generators, with:

- a model registry (registry.py) so new architectures plug in easily
- a config-driven pipeline
- modular math components (probability paths, ODEs, schedules)
- reusable training loop with callbacks
- clean separation of concerns in the repo layout

This project currently includes an implementation of a Conditional UNet 2D trained with classifier-free guidance flow matching on MNIST.

## Repo Structure
```
flow_matching_designs/
│
├── configs/                     # YAML config files for training
│   ├── mnist_baseline.yaml
│   └── mnist_cfg_strong.yaml
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
## 📤 Exporting a Checkpoint
```
python scripts/export_checkpoint.py \
    --checkpoint ckpts/mnist_unet2d.pt \
    --out models/exported_model.pt
```
## 🖼 Sampling Images

Generate MNIST digits from a trained model:
```
python scripts/sample_mnist.py \
    --checkpoint ckpts/mnist_unet2d.pt \
    --n_samples 64 \
    --outfile samples.png
```

This will:

- Load the trained model
- Integrate the flow ODE
- Save generated samples to samples.png

