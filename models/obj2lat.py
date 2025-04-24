"""Object-to-latent regressor for manipulator generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ObjectToLatent(nn.Module):
    """CNN for mapping object voxels to manipulator latent vectors."""
    
    def __init__(self, latent_dim: int = 32):
        """Initialize network.
        
        Args:
            latent_dim: Dimension of target latent space
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Input: (1, 32, 32, 32)
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16, 16)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 8, 8)
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 4, 4, 4)
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 2, 2, 2)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 32, 32, 32)
            
        Returns:
            Predicted latent vectors
        """
        return self.encoder(x)


def triplet_margin_loss(anchor: torch.Tensor,
                       positive: torch.Tensor,
                       negative: torch.Tensor,
                       margin: float = 1.0) -> torch.Tensor:
    """Compute triplet margin loss.
    
    Args:
        anchor: Anchor samples
        positive: Positive samples
        negative: Negative samples
        margin: Margin for triplet loss
        
    Returns:
        Loss value
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


class ObjectLatentLoss(nn.Module):
    """Combined loss for object-to-latent training."""
    
    def __init__(self, margin: float = 1.0):
        """Initialize loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss.
        
        Args:
            pred: Predicted latent vectors
            target: Target latent vectors (positives)
            neg: Negative latent vectors
            
        Returns:
            Tuple of (total_loss, mse_loss)
        """
        mse_loss = F.mse_loss(pred, target)
        triplet_loss = triplet_margin_loss(pred, target, neg, self.margin)
        total_loss = mse_loss + triplet_loss
        return total_loss, mse_loss 