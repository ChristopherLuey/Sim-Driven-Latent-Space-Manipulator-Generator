"""Utility functions for the Sim-Driven Latent-Space Manipulator Generator."""

import os
import torch
import numpy as np
from typing import Tuple, Dict, Any


def sample_random_mask(Nx: int, Ny: int, p: float = 0.3) -> np.ndarray:
    """Generate a random binary mask with given fill probability.
    
    Args:
        Nx: Number of rows
        Ny: Number of columns
        p: Fill probability (default: 0.3)
        
    Returns:
        Binary numpy array of shape (Nx, Ny)
    """
    return (np.random.random((Nx, Ny)) < p).astype(np.float32)


def load_object_voxel(idx: int, size: Tuple[int, int, int]) -> torch.Tensor:
    """Load object voxel grid from disk or generate random one if not found.
    
    Args:
        idx: Object index
        size: (depth, height, width) dimensions
        
    Returns:
        Voxel grid tensor of shape (1, D, H, W)
    """
    path = f"data/raw/obj/{idx}.pt"
    if os.path.exists(path):
        return torch.load(path)
    else:
        # Generate simple random voxel grid for testing
        voxel = torch.rand(1, *size) > 0.8
        return voxel.float()


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   path: str,
                   **kwargs: Any) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   path: str) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        path: Path to checkpoint file
        
    Returns:
        Dictionary containing loaded checkpoint info
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def get_device() -> torch.device:
    """Get PyTorch device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 