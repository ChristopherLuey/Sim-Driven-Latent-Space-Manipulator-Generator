"""Training script for manipulator VAE."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from typing import Tuple

import sys
sys.path.append("..")
from models.manip_vae import ManipulatorVAE
from utils import save_checkpoint, get_device


class ManipulatorVoxelDataset(Dataset):
    """Dataset for manipulator voxel grids."""
    
    def __init__(self, data_dir: str):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing voxel grid files
        """
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir)
                               if f.endswith('.pt')])
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        path = os.path.join(self.data_dir, self.file_list[idx])
        voxel = torch.load(path)
        return voxel


def vae_loss(recon: torch.Tensor,
             target: torch.Tensor,
             mu: torch.Tensor,
             logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss.
    
    Args:
        recon: Reconstructed samples
        target: Target samples
        mu: Mean vectors
        logvar: Log variance vectors
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    recon_loss = nn.BCELoss(reduction='sum')(recon, target)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kld
    
    # Normalize by batch size
    batch_size = target.size(0)
    total_loss = total_loss / batch_size
    recon_loss = recon_loss / batch_size
    kld = kld / batch_size
    
    return total_loss, recon_loss, kld


def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float, float]:
    """Train for one epoch.
    
    Args:
        model: VAE model
        loader: Data loader
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average_total_loss, average_recon_loss, average_kld)
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        recon, mu, logvar = model(batch)
        loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld.item()
    
    return (total_loss / len(loader),
            total_recon / len(loader),
            total_kld / len(loader))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing training data")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=32,
                       help="Latent space dimension")
    parser.add_argument("--ckpt_out", type=str, required=True,
                       help="Path to save checkpoints")
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset and loader
    dataset = ManipulatorVoxelDataset(
        os.path.join(args.data_dir, "manip_vox"))
    loader = DataLoader(dataset,
                       batch_size=args.batch_size,
                       shuffle=True,
                       num_workers=4)
    
    # Create model
    model = ManipulatorVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        avg_loss, avg_recon, avg_kld = train_epoch(
            model, loader, optimizer, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f})")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(args.ckpt_out, "vae_best.pt"),
                recon_loss=avg_recon,
                kld=avg_kld
            )
        
        # Save latest
        save_checkpoint(
            model, optimizer, epoch, avg_loss,
            os.path.join(args.ckpt_out, "vae_latest.pt"),
            recon_loss=avg_recon,
            kld=avg_kld
        )


if __name__ == "__main__":
    main() 