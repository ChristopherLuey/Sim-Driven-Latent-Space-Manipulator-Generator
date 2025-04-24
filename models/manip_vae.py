"""Variational Autoencoder for manipulator generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ManipulatorVAE(nn.Module):
    """VAE for encoding and generating manipulator designs."""
    
    def __init__(self, latent_dim: int = 32):
        """Initialize VAE.
        
        Args:
            latent_dim: Dimension of latent space
        """
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            x = torch.zeros(1, 1, 5, 5, 20)  # Example input
            flat_size = self.encoder(x).shape[1]
        
        # Latent space projections
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, flat_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 2, 2, 5)),  # Reshape to match encoder output
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, 1, D, H, W)
            
        Returns:
            Tuple of (mu, logvar) tensors
        """
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to manipulator designs.
        
        Args:
            z: Latent vectors of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed manipulator designs
        """
        x = self.decoder_input(z)
        return self.decoder(x)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE training.
        
        Args:
            mu: Mean vectors
            logvar: Log variance vectors
            
        Returns:
            Sampled latent vectors
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (batch_size, 1, D, H, W)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample new manipulator designs from prior.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated manipulator designs
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z) 