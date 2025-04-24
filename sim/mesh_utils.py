"""Mesh utilities for converting between voxel and mesh representations."""

import numpy as np
import trimesh
from skimage import measure
from typing import Tuple


def vox_to_mesh(mask: np.ndarray, out_path: str) -> str:
    """Convert a voxel mask to a mesh and save it.
    
    Args:
        mask: Binary voxel grid of shape (N, M) or (D, H, W)
        out_path: Path to save the output mesh
        
    Returns:
        Path to the saved mesh file
    """
    # If 2D mask, extrude to 3D
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=1)
        mask = np.repeat(mask, 5, axis=1)  # Extrude to reasonable thickness
    
    # Pad with zeros for marching cubes
    padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    
    # Generate mesh using marching cubes
    verts, faces, normals, values = measure.marching_cubes(padded, level=0.5)
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)
    
    # Center the mesh
    mesh.vertices -= mesh.center_mass
    
    # Save mesh
    mesh.export(out_path)
    
    return out_path


def get_mesh_bounds(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of a mesh.
    
    Args:
        mesh_path: Path to mesh file
        
    Returns:
        Tuple of (min_bounds, max_bounds) as numpy arrays
    """
    mesh = trimesh.load(mesh_path)
    return mesh.bounds[0], mesh.bounds[1] 