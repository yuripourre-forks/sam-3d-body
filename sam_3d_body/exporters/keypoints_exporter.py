# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import numpy as np
from typing import Optional, Dict, Any
import torch


class KeypointsExporter:
    """
    Exports keypoints and joint data to JSON format.
    This format is designed to be easy to use for creating custom BVH exporters.
    """

    def __init__(self):
        """Initialize the keypoints exporter."""
        # Import keypoint metadata
        from sam_3d_body.metadata.mhr70 import mhr_names, pose_info

        self.keypoint_names = mhr_names  # 70 keypoints
        self.pose_info = pose_info

    def export(
        self,
        pred_keypoints_3d: np.ndarray,
        pred_joint_coords: np.ndarray,
        pred_global_rots: np.ndarray,
        output_path: str,
        image_path: Optional[str] = None,
        pred_keypoints_2d: Optional[np.ndarray] = None,
        pred_cam_t: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None,
        pred_vertices: Optional[np.ndarray] = None,
        include_vertices: bool = False,
    ):
        """
        Export keypoints and joint data to JSON.

        Args:
            pred_keypoints_3d: (70, 3) or (1, 70, 3) - 3D keypoint positions
            pred_joint_coords: (127, 3) or (1, 127, 3) - Joint 3D positions
            pred_global_rots: (127, 3, 3) or (1, 127, 3, 3) or (127, 4) or (1, 127, 4) - Joint rotations (rotation matrices or quaternions)
            output_path: Path to output JSON file
            image_path: Optional path to source image
            pred_keypoints_2d: Optional (70, 2) - 2D keypoint predictions for accuracy computation
            pred_cam_t: Optional (3,) - Camera translation
            focal_length: Optional float - Camera focal length
            pred_vertices: Optional (V, 3) or (1, V, 3) - Mesh vertices
            include_vertices: Whether to include vertices in export (default: False)
        """
        # Convert to numpy if needed
        if isinstance(pred_keypoints_3d, torch.Tensor):
            pred_keypoints_3d = pred_keypoints_3d.detach().cpu().numpy()
        if isinstance(pred_joint_coords, torch.Tensor):
            pred_joint_coords = pred_joint_coords.detach().cpu().numpy()
        if isinstance(pred_global_rots, torch.Tensor):
            pred_global_rots = pred_global_rots.detach().cpu().numpy()
        if pred_keypoints_2d is not None and isinstance(pred_keypoints_2d, torch.Tensor):
            pred_keypoints_2d = pred_keypoints_2d.detach().cpu().numpy()
        if pred_cam_t is not None and isinstance(pred_cam_t, torch.Tensor):
            pred_cam_t = pred_cam_t.detach().cpu().numpy()
        if pred_vertices is not None and isinstance(pred_vertices, torch.Tensor):
            pred_vertices = pred_vertices.detach().cpu().numpy()

        # Handle single frame - add frame dimension if needed
        if pred_keypoints_3d.ndim == 2:
            pred_keypoints_3d = pred_keypoints_3d[None]
        if pred_joint_coords.ndim == 2:
            pred_joint_coords = pred_joint_coords[None]
        if pred_global_rots.ndim == 3:
            # Could be (J, 3, 3) or (J, 4)
            if pred_global_rots.shape[-1] == 3 and pred_global_rots.shape[-2] == 3:
                pred_global_rots = pred_global_rots[None]
            elif pred_global_rots.shape[-1] == 4:
                pred_global_rots = pred_global_rots[None]
        elif pred_global_rots.ndim == 2:
            # (J, 4) quaternions
            pred_global_rots = pred_global_rots[None]
        if pred_keypoints_2d is not None and pred_keypoints_2d.ndim == 2:
            pred_keypoints_2d = pred_keypoints_2d[None]
        if pred_vertices is not None and pred_vertices.ndim == 2:
            pred_vertices = pred_vertices[None]

        num_frames = pred_keypoints_3d.shape[0]
        num_keypoints = pred_keypoints_3d.shape[1]
        num_joints = pred_joint_coords.shape[1]

        # Convert rotations to multiple formats
        import scipy.spatial.transform as transform
        
        if pred_global_rots.shape[-1] == 3 and pred_global_rots.shape[-2] == 3:
            # Input is rotation matrices
            flat_rots = pred_global_rots.reshape(-1, 3, 3)
            r = transform.Rotation.from_matrix(flat_rots)
            quaternions = r.as_quat().reshape(num_frames, num_joints, 4)
            rotation_matrices = pred_global_rots.copy()
        elif pred_global_rots.shape[-1] == 4:
            # Input is quaternions
            quaternions = pred_global_rots.copy()
            flat_quats = quaternions.reshape(-1, 4)
            r = transform.Rotation.from_quat(flat_quats)
            rotation_matrices = r.as_matrix().reshape(num_frames, num_joints, 3, 3)
        else:
            raise ValueError(
                f"pred_global_rots must be rotation matrices (..., 3, 3) or quaternions (..., 4), got shape {pred_global_rots.shape}"
            )

        # Normalize quaternions
        norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        quaternions = quaternions / (norms + 1e-8)

        # Convert to Euler angles (ZXY order, degrees)
        flat_quats = quaternions.reshape(-1, 4)
        r = transform.Rotation.from_quat(flat_quats)
        euler_angles = r.as_euler('ZXY', degrees=True).reshape(num_frames, num_joints, 3)

        # Build export data structure
        export_data = {
            "metadata": {
                "format_version": "1.1",
                "source_image": image_path if image_path else None,
                "coordinate_system": {
                    "description": "Right-handed coordinate system (X-right, Y-up, Z-backward)",
                },
                "keypoints": {
                    "count": num_keypoints,
                    "names": self.keypoint_names,
                    "format": "mhr70",
                },
                "joints": {
                    "count": num_joints,
                    "description": "MHR model joints (127 joints including body, hands, and face)",
                },
                "camera": {
                    "focal_length": float(focal_length) if focal_length is not None else None,
                    "translation": pred_cam_t.tolist() if pred_cam_t is not None else None,
                },
            },
            "frames": [],
        }
        
        if include_vertices and pred_vertices is not None:
            num_vertices = pred_vertices.shape[1] if pred_vertices.ndim >= 2 else 0
            export_data["metadata"]["vertices"] = {
                "count": num_vertices,
                "description": "Mesh vertices from MHR model",
            }

        # Export each frame
        for frame_idx in range(num_frames):
            frame_data = {
                "frame": frame_idx,
                "keypoints_3d": [],
                "keypoints_2d": [],
                "joints": {
                    "positions": [],
                    "rotations": {
                        "quaternion": [],
                        "rotation_matrix": [],
                        "euler_zyx_degrees": [],
                    },
                },
                "root": {
                    "position": pred_joint_coords[frame_idx, 0].tolist(),
                },
            }
            
            # Export vertices if requested
            if include_vertices and pred_vertices is not None:
                frame_data["vertices"] = pred_vertices[frame_idx].tolist()

            # Export 3D keypoints with accuracy metrics
            for kpt_idx in range(num_keypoints):
                kpt_3d = pred_keypoints_3d[frame_idx, kpt_idx]
                depth = float(kpt_3d[2])
                is_visible = depth > 0  # Keypoint is in front of camera if Z > 0
                
                # Depth-based confidence: closer = more confident (normalized to 0-1 range)
                # Assuming reasonable depth range: 0.5m to 10m
                depth_confidence = max(0.0, min(1.0, 1.0 - (depth - 0.5) / 9.5)) if is_visible else 0.0
                
                kpt_data = {
                    "id": kpt_idx,
                    "name": self.keypoint_names[kpt_idx],
                    "position_3d": kpt_3d.tolist(),
                    "depth": depth,
                    "accuracy": {
                        "is_visible": is_visible,
                        "depth_confidence": float(depth_confidence),
                    },
                }
                frame_data["keypoints_3d"].append(kpt_data)

            # Export 2D keypoints if available
            if pred_keypoints_2d is not None:
                for kpt_idx in range(num_keypoints):
                    kpt_2d = pred_keypoints_2d[frame_idx, kpt_idx]
                    kpt_2d_data = {
                        "id": kpt_idx,
                        "name": self.keypoint_names[kpt_idx],
                        "position_2d": kpt_2d.tolist(),
                    }
                    frame_data["keypoints_2d"].append(kpt_2d_data)

            # Export joint positions and rotations in multiple formats
            for joint_idx in range(num_joints):
                joint_pos = pred_joint_coords[frame_idx, joint_idx].tolist()
                joint_quat = quaternions[frame_idx, joint_idx].tolist()
                joint_rotmat = rotation_matrices[frame_idx, joint_idx].tolist()
                joint_euler = euler_angles[frame_idx, joint_idx].tolist()

                frame_data["joints"]["positions"].append(joint_pos)
                frame_data["joints"]["rotations"]["quaternion"].append(joint_quat)
                frame_data["joints"]["rotations"]["rotation_matrix"].append(joint_rotmat)
                frame_data["joints"]["rotations"]["euler_zyx_degrees"].append(joint_euler)

            export_data["frames"].append(frame_data)

        # Write to JSON file
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported keypoints to {output_path} ({num_frames} frame(s))")

