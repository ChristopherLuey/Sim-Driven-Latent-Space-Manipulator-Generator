"""Physics simulator for manipulator grasping using PyBullet."""

import os
import time
import pybullet as p
import pybullet_data
import numpy as np
from typing import Dict, List, Optional


class Simulator:
    """PyBullet-based physics simulator for manipulator grasping."""
    
    def __init__(self, gui: bool = False, gravity: float = -9.81):
        """Initialize simulator.
        
        Args:
            gui: Whether to use GUI or direct mode
            gravity: Gravity acceleration in m/s^2
        """
        # Connect to physics server
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        
        # Add data path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set gravity
        p.setGravity(0, 0, gravity)
        
        # Set real-time simulation
        p.setRealTimeSimulation(0)
        
        # Set default camera
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0]
            )
    
    def __del__(self):
        """Cleanup simulator connection."""
        p.disconnect(self.client)
    
    def run_grasp(self, manip_obj: str, target_obj: str, steps: int) -> Dict[str, float]:
        """Run grasping simulation.
        
        Args:
            manip_obj: Path to manipulator mesh/URDF
            target_obj: Path to target object mesh/URDF
            steps: Number of simulation steps
            
        Returns:
            Dictionary with simulation metrics
        """
        # Reset simulation
        p.resetSimulation()
        
        # Load ground plane
        plane_id = p.loadURDF("plane.urdf")
        
        # Load manipulator
        manip_pos = [0, 0, 0.1]
        manip_orn = p.getQuaternionFromEuler([0, 0, 0])
        manip_id = p.loadURDF(manip_obj, manip_pos, manip_orn, useFixedBase=True)
        
        # Load target object
        obj_pos = [0.2, 0, 0.1]
        obj_orn = p.getQuaternionFromEuler([0, 0, 0])
        obj_id = p.loadURDF(target_obj, obj_pos, obj_orn)
        
        # Run simulation
        contact_points: List[int] = []
        total_force = 0.0
        
        for _ in range(steps):
            p.stepSimulation()
            
            # Get contact points
            points = p.getContactPoints(manip_id, obj_id)
            if points:
                contact_points.append(len(points))
                total_force += sum(pt[9] for pt in points)  # Normal force
        
        # Compute metrics
        avg_contacts = np.mean(contact_points) if contact_points else 0
        avg_force = total_force / steps if total_force > 0 else 0
        
        return {
            "contact_count": avg_contacts,
            "force": avg_force
        }
    
    def get_object_pose(self, obj_id: int) -> tuple:
        """Get current pose of an object.
        
        Args:
            obj_id: PyBullet body ID
            
        Returns:
            Tuple of (position, orientation)
        """
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        return pos, orn 