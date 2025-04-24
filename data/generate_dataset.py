"""Dataset generation script for manipulator training."""

import os
import csv
import torch
import numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, Any

import sys
sys.path.append("..")
from sim.mesh_utils import vox_to_mesh
from sim.simulator import Simulator
from utils import sample_random_mask, load_object_voxel


def generate_sample(idx: int,
                   output_dir: str,
                   sim_steps: int,
                   simulator: Simulator) -> Dict[str, Any]:
    """Generate a single training sample.
    
    Args:
        idx: Sample index
        output_dir: Output directory
        sim_steps: Number of simulation steps
        simulator: Physics simulator instance
        
    Returns:
        Dictionary with sample metrics
    """
    # Create directories if they don't exist
    os.makedirs(f"{output_dir}/manip", exist_ok=True)
    os.makedirs(f"{output_dir}/manip_vox", exist_ok=True)
    os.makedirs(f"{output_dir}/obj", exist_ok=True)
    
    # Generate random manipulator
    voxel_manip = sample_random_mask(5, 20)
    voxel_manip = np.expand_dims(voxel_manip, axis=0)  # Add channel dimension
    
    # Save manipulator voxel grid
    torch.save(torch.from_numpy(voxel_manip),
              f"{output_dir}/manip_vox/{idx}.pt")
    
    # Convert to mesh and save
    manip_mesh_path = vox_to_mesh(voxel_manip[0],
                                 f"{output_dir}/manip/{idx}.obj")
    
    # Load or generate object
    obj_voxel = load_object_voxel(idx, (32, 32, 32))
    torch.save(obj_voxel, f"{output_dir}/obj/{idx}.pt")
    
    # Convert to mesh and save
    obj_mesh_path = vox_to_mesh(obj_voxel[0, 0].numpy(),
                               f"{output_dir}/obj/{idx}.obj")
    
    # Run simulation
    metrics = simulator.run_grasp(manip_mesh_path, obj_mesh_path, sim_steps)
    
    return {
        "index": idx,
        "contact_count": metrics["contact_count"],
        "force": metrics["force"]
    }


def main():
    """Main dataset generation function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./data/raw",
                       help="Output directory")
    parser.add_argument("--sim_steps", type=int, default=300,
                       help="Number of simulation steps")
    args = parser.parse_args()
    
    # Create simulator
    simulator = Simulator(gui=False)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup CSV file
    csv_path = f"{args.output_dir}/metrics.csv"
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["index", "contact_count", "force"])
        if not csv_exists:
            writer.writeheader()
        
        # Generate samples
        for idx in tqdm(range(args.n_samples)):
            metrics = generate_sample(idx, args.output_dir,
                                   args.sim_steps, simulator)
            writer.writerow(metrics)


if __name__ == "__main__":
    main() 