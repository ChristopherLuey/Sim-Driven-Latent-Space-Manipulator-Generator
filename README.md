# Sim-Driven Latent-Space Manipulator Generator

A PyTorch-based project for generating robot manipulator designs optimized for specific objects using simulation-driven latent space learning.

## Overview

This project implements a system that can generate custom manipulator designs for grasping specific objects. It uses:

1. A variational autoencoder (VAE) to learn a latent space of manipulator designs
2. A physics simulator to evaluate grasp quality
3. A neural network regressor to map object shapes to optimal manipulator latent vectors

## Installation

```bash
# Clone the repository
git clone https://github.com/christopherluey/Sim-Driven-Latent-Space-Manipulator-Generator.git
cd Sim-Driven-Latent-Space-Manipulator-Generator

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/
│   ├── generate_dataset.py  # Dataset generation script
│   └── raw/                 # Generated dataset
├── sim/
│   ├── simulator.py         # PyBullet physics simulator
│   └── mesh_utils.py        # Mesh conversion utilities
├── models/
│   ├── manip_vae.py        # Manipulator VAE model
│   └── obj2lat.py          # Object-to-latent regressor
├── train/
│   ├── train_vae.py        # VAE training script
│   └── train_regressor.py  # Regressor training script
├── infer.py                # Inference script
├── utils.py                # Utility functions
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Usage

### 1. Generate Training Data

```bash
python data/generate_dataset.py \
    --n_samples 1000 \
    --output_dir ./data/raw \
    --sim_steps 300
```

### 2. Train the VAE

```bash
python train/train_vae.py \
    --data_dir ./data/raw \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --latent_dim 32 \
    --ckpt_out ./checkpoints
```

### 3. Train the Regressor

```bash
python train/train_regressor.py \
    --data_dir ./data/raw \
    --vae_ckpt ./checkpoints/vae_best.pt \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --margin 1.0 \
    --ckpt_out ./checkpoints
```

### 4. Generate Manipulators

```bash
python infer.py \
    --obj_path path/to/object.pt \
    --vae_ckpt ./checkpoints/vae_best.pt \
    --reg_ckpt ./checkpoints/regressor_best.pt \
    --out_voxel output.pt \
    --out_mesh output.obj
```

## Data Format

- Object voxels: Binary 3D grids of shape `(1, 32, 32, 32)`
- Manipulator voxels: Binary 2D grids of shape `(1, 5, 20)`
- All voxel grids are saved as PyTorch tensors (`.pt` files)
- Meshes are saved in OBJ format

## Training Process

1. **Dataset Generation**:
   - Random manipulator designs are generated
   - Physics simulation evaluates grasp quality
   - Data is saved with quality metrics

2. **VAE Training**:
   - Learns to encode/decode manipulator designs
   - Uses reconstruction loss + KL divergence

3. **Regressor Training**:
   - Maps object shapes to manipulator latent vectors
   - Uses MSE loss + triplet margin loss
   - Triplet mining based on grasp quality

## License

MIT License 