"""Training script for object-to-latent regressor."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from typing import Tuple, List

import sys
sys.path.append("..")
from models.manip_vae import ManipulatorVAE
from models.obj2lat import ObjectToLatent, ObjectLatentLoss
from utils import save_checkpoint, load_checkpoint, get_device


class ObjectManipulatorDataset(Dataset):
    """Dataset for object-manipulator pairs."""
    
    def __init__(self, data_dir: str, metrics_file: str):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing data
            metrics_file: Path to metrics CSV file
        """
        self.data_dir = data_dir
        
        # Load metrics and sort by grasp quality
        metrics = pd.read_csv(metrics_file)
        metrics['quality'] = metrics['contact_count'] * metrics['force']
        self.metrics = metrics.sort_values('quality', ascending=False)
        
        # Get file paths
        self.obj_files = [
            os.path.join(data_dir, "obj", f"{idx}.pt")
            for idx in self.metrics.index
        ]
        self.manip_files = [
            os.path.join(data_dir, "manip_vox", f"{idx}.pt")
            for idx in self.metrics.index
        ]
    
    def __len__(self) -> int:
        return len(self.metrics)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (object_voxel, positive_manip, negative_manip)
        """
        # Load object
        obj = torch.load(self.obj_files[idx])
        
        # Load positive manipulator (current index)
        pos_manip = torch.load(self.manip_files[idx])
        
        # Load negative manipulator (random from bottom 50%)
        neg_idx = np.random.randint(len(self) // 2, len(self))
        neg_manip = torch.load(self.manip_files[neg_idx])
        
        return obj, pos_manip, neg_manip


def train_epoch(regressor: nn.Module,
                vae: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        regressor: Regressor model
        vae: VAE model
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average_total_loss, average_mse_loss)
    """
    regressor.train()
    vae.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    
    for obj, pos, neg in tqdm(loader, desc="Training", leave=False):
        obj = obj.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        
        optimizer.zero_grad()
        
        # Get target latent vectors
        with torch.no_grad():
            pos_mu, _ = vae.encode(pos)
            neg_mu, _ = vae.encode(neg)
        
        # Get predicted latent vector
        pred = regressor(obj)
        
        # Compute loss
        loss, mse = criterion(pred, pos_mu, neg_mu)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
    
    return (total_loss / len(loader),
            total_mse / len(loader))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing training data")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                       help="Path to VAE checkpoint")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Triplet loss margin")
    parser.add_argument("--ckpt_out", type=str, required=True,
                       help="Path to save checkpoints")
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load VAE
    vae = ManipulatorVAE().to(device)
    load_checkpoint(vae, None, args.vae_ckpt)
    vae.eval()
    
    # Create dataset and loader
    dataset = ObjectManipulatorDataset(
        args.data_dir,
        os.path.join(args.data_dir, "metrics.csv")
    )
    loader = DataLoader(dataset,
                       batch_size=args.batch_size,
                       shuffle=True,
                       num_workers=4)
    
    # Create model and optimizer
    regressor = ObjectToLatent(latent_dim=vae.latent_dim).to(device)
    optimizer = optim.Adam(regressor.parameters(), lr=args.lr)
    criterion = ObjectLatentLoss(margin=args.margin)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        avg_loss, avg_mse = train_epoch(
            regressor, vae, loader, criterion, optimizer, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f})")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                regressor, optimizer, epoch, avg_loss,
                os.path.join(args.ckpt_out, "regressor_best.pt"),
                mse_loss=avg_mse
            )
        
        # Save latest
        save_checkpoint(
            regressor, optimizer, epoch, avg_loss,
            os.path.join(args.ckpt_out, "regressor_latest.pt"),
            mse_loss=avg_mse
        )


if __name__ == "__main__":
    main() 