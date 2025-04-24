"""Inference script for generating manipulator designs."""

import os
import torch
import argparse
from typing import Optional

from models.manip_vae import ManipulatorVAE
from models.obj2lat import ObjectToLatent
from utils import load_checkpoint, get_device
from sim.mesh_utils import vox_to_mesh


def load_object_voxel(path: str) -> torch.Tensor:
    """Load object voxel grid from file.
    
    Args:
        path: Path to object file (supports .pt or .obj)
        
    Returns:
        Object voxel grid tensor
    """
    if path.endswith('.pt'):
        return torch.load(path)
    else:
        raise NotImplementedError(
            "Only .pt voxel files supported for now. "
            "Add mesh loading support if needed.")


def generate_manipulator(obj_path: str,
                        vae_ckpt: str,
                        reg_ckpt: str,
                        out_voxel: Optional[str] = None,
                        out_mesh: Optional[str] = None,
                        device: Optional[torch.device] = None) -> torch.Tensor:
    """Generate manipulator design for object.
    
    Args:
        obj_path: Path to object file
        vae_ckpt: Path to VAE checkpoint
        reg_ckpt: Path to regressor checkpoint
        out_voxel: Path to save voxel grid (optional)
        out_mesh: Path to save mesh (optional)
        device: Device to run on (optional)
        
    Returns:
        Generated manipulator voxel grid
    """
    if device is None:
        device = get_device()
    
    # Load models
    vae = ManipulatorVAE().to(device)
    regressor = ObjectToLatent(latent_dim=vae.latent_dim).to(device)
    
    load_checkpoint(vae, None, vae_ckpt)
    load_checkpoint(regressor, None, reg_ckpt)
    
    vae.eval()
    regressor.eval()
    
    # Load object
    obj_voxel = load_object_voxel(obj_path).to(device)
    
    # Generate manipulator
    with torch.no_grad():
        latent = regressor(obj_voxel)
        manip_voxel = vae.decode(latent)
    
    # Save outputs
    if out_voxel:
        torch.save(manip_voxel.cpu(), out_voxel)
    
    if out_mesh:
        vox_to_mesh(manip_voxel[0, 0].cpu().numpy(), out_mesh)
    
    return manip_voxel


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True,
                       help="Path to object file")
    parser.add_argument("--vae_ckpt", type=str, required=True,
                       help="Path to VAE checkpoint")
    parser.add_argument("--reg_ckpt", type=str, required=True,
                       help="Path to regressor checkpoint")
    parser.add_argument("--out_voxel", type=str, default=None,
                       help="Path to save voxel grid")
    parser.add_argument("--out_mesh", type=str, default=None,
                       help="Path to save mesh")
    args = parser.parse_args()
    
    # Generate manipulator
    manip_voxel = generate_manipulator(
        args.obj_path,
        args.vae_ckpt,
        args.reg_ckpt,
        args.out_voxel,
        args.out_mesh
    )
    
    print(f"Generated manipulator shape: {manip_voxel.shape}")
    if args.out_voxel:
        print(f"Saved voxel grid to: {args.out_voxel}")
    if args.out_mesh:
        print(f"Saved mesh to: {args.out_mesh}")


if __name__ == "__main__":
    main() 