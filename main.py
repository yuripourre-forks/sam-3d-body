# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import mhr_names
from tqdm import tqdm
import scipy.spatial.transform as transform

EPSILON = 1e-6


def rotmat_to_euler_zyx(rotmat):
    """
    Convert rotation matrix to Euler angles in ZYX order (BVH standard).
    Args:
        rotmat: (3, 3) rotation matrix
    Returns:
        euler: (3,) Euler angles in degrees [Z, Y, X]
    """
    if isinstance(rotmat, torch.Tensor):
        rotmat = rotmat.cpu().numpy()
    if rotmat.shape == (3, 3):
        rotmat = rotmat.reshape(1, 3, 3)
    r = transform.Rotation.from_matrix(rotmat)
    euler = r.as_euler('ZYX', degrees=True)
    if euler.ndim == 2:
        euler = euler[0]
    return euler


def compute_bone_length(parent_pos, child_pos):
    """Compute bone length between parent and child joints."""
    return np.linalg.norm(child_pos - parent_pos)


def get_keypoint_index(name):
    """Get index of keypoint by name."""
    try:
        return mhr_names.index(name)
    except ValueError:
        # Try with underscores instead of hyphens
        name_alt = name.replace("-", "_")
        for i, n in enumerate(mhr_names):
            if n.replace("-", "_") == name_alt:
                return i
    return None


class BVHExporter:
    """
    Exports 3D skeleton data to BVH format using MHR model's keypoint positions.

    Hierarchy Structure (Standard Humanoid):
    Hips (root)
    ├── Spine
    │   └── Spine1
    │       ├── Neck
    │       │   └── Head (end site)
    │       ├── LeftShoulder
    │       │   └── LeftArm
    │       │       └── LeftForeArm
    │       │           └── LeftHand (end site)
    │       └── RightShoulder
    │           └── RightArm
    │               └── RightForeArm
    │                   └── RightHand (end site)
    ├── LeftUpLeg
    │   └── LeftLeg
    │       └── LeftFoot
    │           └── LeftToeBase (end site)
    └── RightUpLeg
        └── RightLeg
            └── RightFoot
                └── RightToeBase (end site)

    This matches the standard Mixamo/Unity humanoid skeleton structure.
    """

    def __init__(self):
        # Define skeleton hierarchy matching the exact target BVH structure
        # Format: (joint_name, parent_name, joint_coord_idx, keypoint_name, [channels])
        # This matches the standard Mixamo/Unity humanoid skeleton structure

        self.skeleton_hierarchy = [
            # Root (computed as hip center)
            ("Hips", None, None, None, ["Xposition", "Yposition", "Zposition", "Zrotation", "Yrotation", "Xrotation"]),

            # Spine chain - THREE children from Hips: Spine, LeftUpLeg, RightUpLeg
            ("Spine", "Hips", None, None, ["Zrotation", "Yrotation", "Xrotation"]),
            ("Spine1", "Spine", None, None, ["Zrotation", "Yrotation", "Xrotation"]),

            # THREE children from Spine1: Neck, LeftShoulder, RightShoulder
            ("Neck", "Spine1", None, "neck", ["Zrotation", "Yrotation", "Xrotation"]),
            ("Head", "Neck", None, None, ["Zrotation", "Yrotation", "Xrotation"]),

            # Left arm chain (child of Spine1)
            ("LeftShoulder", "Spine1", None, "left-shoulder", ["Zrotation", "Yrotation", "Xrotation"]),
            ("LeftArm", "LeftShoulder", None, "left-elbow", ["Zrotation", "Yrotation", "Xrotation"]),
            ("LeftForeArm", "LeftArm", None, "left-wrist", ["Zrotation", "Yrotation", "Xrotation"]),
            ("LeftHand", "LeftForeArm", None, None, ["Zrotation", "Yrotation", "Xrotation"]),

            # Right arm chain (child of Spine1)
            ("RightShoulder", "Spine1", None, "right-shoulder", ["Zrotation", "Yrotation", "Xrotation"]),
            ("RightArm", "RightShoulder", None, "right-elbow", ["Zrotation", "Yrotation", "Xrotation"]),
            ("RightForeArm", "RightArm", None, "right-wrist", ["Zrotation", "Yrotation", "Xrotation"]),
            ("RightHand", "RightForeArm", None, None, ["Zrotation", "Yrotation", "Xrotation"]),

            # Left leg chain (child of Hips)
            ("LeftUpLeg", "Hips", None, "left-hip", ["Zrotation", "Yrotation", "Xrotation"]),
            ("LeftLeg", "LeftUpLeg", None, "left-knee", ["Zrotation", "Yrotation", "Xrotation"]),
            ("LeftFoot", "LeftLeg", None, "left-ankle", ["Zrotation", "Yrotation", "Xrotation"]),
            ("LeftToeBase", "LeftFoot", None, "left-big-toe-tip", ["Zrotation", "Yrotation", "Xrotation"]),

            # Right leg chain (child of Hips)
            ("RightUpLeg", "Hips", None, "right-hip", ["Zrotation", "Yrotation", "Xrotation"]),
            ("RightLeg", "RightUpLeg", None, "right-knee", ["Zrotation", "Yrotation", "Xrotation"]),
            ("RightFoot", "RightLeg", None, "right-ankle", ["Zrotation", "Yrotation", "Xrotation"]),
            ("RightToeBase", "RightFoot", None, "right-big-toe-tip", ["Zrotation", "Yrotation", "Xrotation"]),
        ]
        self.joint_map = {joint[0]: joint for joint in self.skeleton_hierarchy}

        # Define End Sites for leaf nodes
        # Note: We'll compute End Sites based on parent->child direction
        # rather than using specific tip keypoints to avoid lateral offsets
        self.end_sites = {
            "Head": None,  # Extend in neck->nose direction
            "LeftHand": "left-middle-tip",
            "RightHand": "right-middle-tip",
            "LeftToeBase": None,  # Extend in ankle->toe direction
            "RightToeBase": None,  # Extend in ankle->toe direction
        }

    def compute_hip_center(self, keypoints_3d):
        """Compute hip center as average of left and right hip."""
        left_hip_idx = get_keypoint_index("left-hip")
        right_hip_idx = get_keypoint_index("right-hip")
        if left_hip_idx is not None and right_hip_idx is not None:
            return (keypoints_3d[left_hip_idx] + keypoints_3d[right_hip_idx]) / 2.0
        return np.array([0.0, 0.0, 0.0])

    def align_to_y_up(self, keypoints_3d, joint_rotations=None):
        """
        Rotate keypoints and rotations to align the spine (Hips -> Neck) with +Y axis.
        Also fixes 'upside down' issues.

        Returns:
            keypoints_3d: Aligned keypoints
            joint_rotations: Aligned rotation matrices
            alignment_matrix: The rotation matrix used for alignment
        """
        # Find Hips and Neck
        hips_center = self.compute_hip_center(keypoints_3d)
        neck_idx = get_keypoint_index("neck")

        alignment_matrix = np.eye(3)

        if neck_idx is not None:
            neck_pos = keypoints_3d[neck_idx]
            spine_vec = neck_pos - hips_center

            # Normalize
            norm = np.linalg.norm(spine_vec)
            if norm < EPSILON:
                return keypoints_3d, joint_rotations, alignment_matrix
            spine_vec /= norm

            # Target vector is +Y (0, 1, 0)
            target_vec = np.array([0.0, 1.0, 0.0])

            # Rotation to align spine_vec to target_vec
            axis = np.cross(spine_vec, target_vec)
            axis_norm = np.linalg.norm(axis)

            if axis_norm < EPSILON:
                # Already aligned or opposite
                if np.dot(spine_vec, target_vec) < 0:
                    # Opposite, rotate 180 around X
                    alignment_matrix = transform.Rotation.from_euler('x', 180, degrees=True).as_matrix()
                else:
                    return keypoints_3d, joint_rotations, alignment_matrix
            else:
                axis /= axis_norm
                angle = np.arccos(np.clip(np.dot(spine_vec, target_vec), -1.0, 1.0))
                alignment_matrix = transform.Rotation.from_rotvec(axis * angle).as_matrix()

            # Apply rotation to all keypoints
            keypoints_3d = np.dot(keypoints_3d, alignment_matrix.T)

            # Apply rotation to joint rotations if available
            # R_new = R_align * R_old
            if joint_rotations is not None:
                joint_rotations = np.matmul(alignment_matrix, joint_rotations)

        return keypoints_3d, joint_rotations, alignment_matrix

    def get_joint_position(self, joint_name, keypoints_3d, hip_center):
        """Get 3D position for a joint."""
        if joint_name == "Hips":
            return hip_center

        # Spine has same position as Hips (zero offset)
        if joint_name == "Spine":
            return hip_center

        # Special handling for intermediate spine joints
        if joint_name == "Spine1":
            # Spine1 is between Hips and Neck (approximately chest level)
            neck_idx = get_keypoint_index("neck")
            if neck_idx is not None:
                neck_pos = keypoints_3d[neck_idx]
                # Position Spine1 between hips and neck
                SPINE1_INTERPOLATION = 0.5
                return hip_center + SPINE1_INTERPOLATION * (neck_pos - hip_center)
            return hip_center

        # Head position computed from eyes and ears for better orientation
        if joint_name == "Head":
            left_eye_idx = get_keypoint_index("left-eye")
            right_eye_idx = get_keypoint_index("right-eye")
            left_ear_idx = get_keypoint_index("left-ear")
            right_ear_idx = get_keypoint_index("right-ear")

            positions = []
            for idx in [left_eye_idx, right_eye_idx, left_ear_idx, right_ear_idx]:
                if idx is not None and idx < len(keypoints_3d):
                    positions.append(keypoints_3d[idx])

            if len(positions) >= 2:
                return np.mean(positions, axis=0)
            else:
                # Fallback to nose
                nose_idx = get_keypoint_index("nose")
                if nose_idx is not None:
                    return keypoints_3d[nose_idx]

        # Hands have same position as wrists (zero offset)
        if joint_name in ["LeftHand", "RightHand"]:
            wrist_name = "left-wrist" if "Left" in joint_name else "right-wrist"
            wrist_idx = get_keypoint_index(wrist_name)
            if wrist_idx is not None:
                return keypoints_3d[wrist_idx]

        # Standard keypoint lookup
        joint_info = self.joint_map.get(joint_name)
        if joint_info and joint_info[3]:  # has keypoint_name
            idx = get_keypoint_index(joint_info[3])
            if idx is not None and idx < len(keypoints_3d):
                return keypoints_3d[idx]

        # For other intermediate joints, interpolate between parent and child
        parent_name = joint_info[1] if joint_info else None
        if parent_name:
            parent_pos = self.get_joint_position(parent_name, keypoints_3d, hip_center)
            # Get first child with a keypoint
            children = [j[0] for j in self.skeleton_hierarchy if j[1] == joint_name]
            for child_name in children:
                child_info = self.joint_map.get(child_name)
                if child_info and child_info[3]:  # Child has keypoint
                    child_pos = self.get_joint_position(child_name, keypoints_3d, hip_center)
                    if parent_pos is not None and child_pos is not None:
                        INTERPOLATION_FACTOR = 0.3
                        return parent_pos + INTERPOLATION_FACTOR * (child_pos - parent_pos)
            # If no child with keypoint, return parent position
            return parent_pos
        return None

    def compute_bone_offset(self, parent_name, child_name, keypoints_3d, hip_center):
        """Compute bone offset from parent to child directly (for pose-as-offset BVH)."""
        parent_pos = self.get_joint_position(parent_name, keypoints_3d, hip_center)
        child_pos = self.get_joint_position(child_name, keypoints_3d, hip_center)

        if parent_pos is not None and child_pos is not None:
            # Direct offset in global space (since we're using zero rotations)
            return child_pos - parent_pos

        # Fallback
        DEFAULT_OFFSET_Y = 10.0
        return np.array([0.0, DEFAULT_OFFSET_Y, 0.0])

    def build_global_rotations_dict(self, keypoints_3d, hip_center, joint_rotations=None):
        """
        Build a dictionary mapping joint names to global rotation matrices.

        For static pose export (pose-as-offset), we use identity rotations.
        The pose is encoded in the bone offsets instead of rotations.

        Args:
            keypoints_3d: 3D keypoint positions (70,3)
            hip_center: Hip center position
            joint_rotations: Optional MHR joint rotation matrices (127, 3, 3) - unused for now

        Returns:
            Dict mapping joint names to identity rotation matrices
        """
        global_rotations = {}

        # Use identity rotations for all joints (pose-as-offset approach)
        for joint_name, parent_name, joint_coord_idx, keypoint_name, channels in self.skeleton_hierarchy:
            global_rotations[joint_name] = np.eye(3)

        return global_rotations

    def get_joint_rotation(self, joint_name, global_rotations):
        """
        Get local rotation for a joint in BVH format (Euler ZYX degrees).

        For static pose export using pose-as-offset approach, all rotations are zero.
        The pose is fully defined by the bone offsets.

        Args:
            joint_name: Name of the joint
            global_rotations: Dict of global rotation matrices (unused for zero rotations)

        Returns:
            Zero Euler angles in ZYX order (degrees)
        """
        return np.array([0.0, 0.0, 0.0])

    def export_to_bvh(self, keypoints_3d, joint_rotations, output_path, frame_time=0.033333):
        """
        Export skeleton to BVH format.

        Args:
            keypoints_3d: (70, 3) array of 3D keypoint positions
            joint_rotations: (127, 3, 3) array of rotation matrices, or None
            output_path: Path to output BVH file
            frame_time: Time per frame in seconds (default 30 FPS)
        """
        hip_center = self.compute_hip_center(keypoints_3d)

        # Align to Y-up
        keypoints_3d, joint_rotations, alignment_matrix = self.align_to_y_up(keypoints_3d, joint_rotations)
        hip_center = self.compute_hip_center(keypoints_3d)

        # Build global rotations dictionary (identity for pose-as-offset)
        global_rotations = self.build_global_rotations_dict(keypoints_3d, hip_center, joint_rotations)

        # Compute bone offsets directly from keypoint positions
        joint_offsets = {}
        for joint_name, parent_name, _, _, _ in self.skeleton_hierarchy:
            if parent_name is None:  # Root (Hips)
                joint_offsets[joint_name] = np.array([0.0, 0.0, 0.0])
            else:
                offset = self.compute_bone_offset(parent_name, joint_name, keypoints_3d, hip_center)
                joint_offsets[joint_name] = offset

        # Write BVH file
        with open(output_path, 'w') as f:
            # Write header
            f.write("HIERARCHY\n")
            # Write skeleton hierarchy starting from root
            self._write_joint(f, "Hips", joint_offsets, 0, keypoints_3d, hip_center)
            f.write("\n")
            f.write("MOTION\n")
            f.write(f"Frames: 1\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            # Write frame data
            frame_data = []
            for joint_name, _, _, _, channels in self.skeleton_hierarchy:
                if joint_name == "Hips":
                    # Root position
                    pos = self.get_joint_position("Hips", keypoints_3d, hip_center)
                    frame_data.extend([pos[0], pos[1], pos[2]])

                # Rotation
                rot = self.get_joint_rotation(joint_name, global_rotations)

                if "Zrotation" in channels:
                    frame_data.append(rot[0])
                if "Yrotation" in channels:
                    frame_data.append(rot[1])
                if "Xrotation" in channels:
                    frame_data.append(rot[2])

            # Write frame data
            f.write(" ".join(f"{val:.6f}" for val in frame_data))
            f.write("\n")

    def _write_joint(self, f, joint_name, joint_offsets, indent_level, keypoints_3d, hip_center):
        """Recursively write joint hierarchy."""
        indent = "  " * indent_level
        joint_info = self.joint_map[joint_name]
        _, parent_name, _, _, channels = joint_info

        if parent_name is None:  # Root
            f.write(f"{indent}ROOT {joint_name}\n")
            f.write(f"{indent}{{\n")
            offset = joint_offsets[joint_name]
            f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
            f.write(f"{indent}  CHANNELS {len(channels)} {' '.join(channels)}\n")
        else:
            offset = joint_offsets[joint_name]
            f.write(f"{indent}JOINT {joint_name}\n")
            f.write(f"{indent}{{\n")
            f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
            f.write(f"{indent}  CHANNELS {len(channels)} {' '.join(channels)}\n")

        # Write children
        children = [j[0] for j in self.skeleton_hierarchy if j[1] == joint_name]
        if children:
            for child_name in children:
                self._write_joint(f, child_name, joint_offsets, indent_level + 1, keypoints_3d, hip_center)
        else:
            # Leaf node - add End Site if defined
            if joint_name in self.end_sites:
                tip_name = self.end_sites[joint_name]
                joint_pos = self.get_joint_position(joint_name, keypoints_3d, hip_center)
                parent_pos = self.get_joint_position(parent_name, keypoints_3d, hip_center)

                if tip_name is None:
                    # Compute End Site by extending in parent->joint direction
                    if joint_pos is not None and parent_pos is not None:
                        direction = joint_pos - parent_pos
                        norm = np.linalg.norm(direction)
                        if norm > EPSILON:
                            # Extension factors for different bones
                            if "Toe" in joint_name:
                                TOE_EXTENSION_FACTOR = 0.3
                                end_offset = direction * TOE_EXTENSION_FACTOR
                            elif joint_name == "Head":
                                HEAD_EXTENSION_FACTOR = 0.5
                                end_offset = direction * HEAD_EXTENSION_FACTOR
                            else:
                                DEFAULT_EXTENSION_FACTOR = 0.2
                                end_offset = direction * DEFAULT_EXTENSION_FACTOR
                        else:
                            DEFAULT_ENDSITE_OFFSET = 5.0
                            end_offset = np.array([0.0, DEFAULT_ENDSITE_OFFSET, 0.0])
                    else:
                        DEFAULT_ENDSITE_OFFSET = 5.0
                        end_offset = np.array([0.0, DEFAULT_ENDSITE_OFFSET, 0.0])
                else:
                    # Use specific tip keypoint (for hands)
                    tip_idx = get_keypoint_index(tip_name)
                    if tip_idx is not None and joint_pos is not None:
                        tip_pos = keypoints_3d[tip_idx]
                        end_offset = tip_pos - joint_pos
                    else:
                        DEFAULT_ENDSITE_OFFSET = 5.0
                        end_offset = np.array([0.0, DEFAULT_ENDSITE_OFFSET, 0.0])

                # Write End Site
                f.write(f"{indent}  End Site\n")
                f.write(f"{indent}  {{\n")
                f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
                f.write(f"{indent}  }}\n")
            else:
                # Generic leaf without End Site definition
                f.write(f"{indent}  End Site\n")
                f.write(f"{indent}  {{\n")
                f.write(f"{indent}    OFFSET 0.000000 0.000000 0.000000\n")
                f.write(f"{indent}  }}\n")

        # Close joint
        f.write(f"{indent}}}\n")


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    # Initialize BVH exporter
    bvh_exporter = BVHExporter()

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        if len(outputs) == 0:
            print(f"No humans detected in {image_path}, skipping...")
            continue

        # Process each detected human
        for idx, output in enumerate(outputs):
            keypoints_3d = output["pred_keypoints_3d"]  # (70, 3)
            joint_rotations = output.get("pred_global_rots")  # (127, 3, 3) or None

            # Convert to numpy if needed
            if isinstance(keypoints_3d, torch.Tensor):
                keypoints_3d = keypoints_3d.cpu().numpy()
            if joint_rotations is not None and isinstance(joint_rotations, torch.Tensor):
                joint_rotations = joint_rotations.cpu().numpy()

            # Generate output filename
            base_name = os.path.basename(image_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            if len(outputs) > 1:
                bvh_filename = f"{base_name_no_ext}_person{idx}.bvh"
            else:
                bvh_filename = f"{base_name_no_ext}.bvh"
            bvh_path = os.path.join(output_folder, bvh_filename)

            # Export to BVH
            try:
                bvh_exporter.export_to_bvh(keypoints_3d, joint_rotations, bvh_path)
                print(f"Exported BVH to {bvh_path}")
            except Exception as e:
                print(f"Error exporting BVH for {image_path}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body to BVH Exporter - Convert 3D skeleton to BVH format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
python main.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

Environment Variables:
SAM3D_MHR_PATH: Path to MHR asset
SAM3D_DETECTOR_PATH: Path to human detection model folder
SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
SAM3D_FOV_PATH: Path to fov estimation model folder
""",
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    args = parser.parse_args()

    main(args)
