
import os
import torch
import numpy as np
import scipy.spatial.transform as transform

class BVHExporter:
    MHR_TO_MIXAMO = {
        "root": "Hips",
        "l_upleg": "LeftUpLeg",
        "l_lowleg": "LeftLeg",
        "l_foot": "LeftFoot",
        "l_talocrural": "LeftFoot",
        "l_toes": "LeftToeBase",
        "l_ball": "LeftToeBase",
        "r_upleg": "RightUpLeg",
        "r_lowleg": "RightLeg",
        "r_foot": "RightFoot",
        "r_talocrural": "RightFoot",
        "r_toes": "RightToeBase",
        "r_ball": "RightToeBase",
        "c_spine0": "Spine02",
        "c_spine1": "Spine01",
        "c_spine2": "Spine",
        "c_spine3": "Spine",
        "c_neck": "neck",
        "c_head": "Head",
        "l_clavicle": "LeftShoulder",
        "l_uparm": "LeftArm",
        "l_lowarm": "LeftForeArm",
        "l_wrist": "LeftHand",
        "r_clavicle": "RightShoulder",
        "r_uparm": "RightArm",
        "r_lowarm": "RightForeArm",
        "r_wrist": "RightHand",
    }

    def __init__(self, model_path=None, model_instance=None, target_skeleton_path=None):
        """
        Initialize the BVH exporter.

        Args:
            model_path: Path to the mhr_model.pt file.
            model_instance: Pre-loaded MHR model instance.
            target_skeleton_path: Optional path to a BVH file defining the target skeleton (rest pose).
        """
        self.model_path = model_path
        self.model = model_instance

        # Model skeleton (Input structure)
        self.model_joint_names = []
        self.model_joint_parents = []
        self.model_rest_quats = []  # Rest pose global quaternions from MHR model

        # Export skeleton (Output structure)
        self.joint_names = []
        self.joint_parents = []
        self.rest_offsets = [] # Offsets in parent frame

        self._load_and_extract_skeleton()

        if target_skeleton_path:
             self._load_target_skeleton(target_skeleton_path)

    def _load_and_extract_skeleton(self):
        if self.model is None:
            if self.model_path is None:
                raise ValueError("Either model_path or model_instance must be provided")
            print(f"Loading MHR model from {self.model_path}...")
            try:
                self.model = torch.jit.load(self.model_path, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Failed to load MHR model: {e}")

        model = self.model

        # Extract hierarchy from model
        try:
            character_torch = model.character_torch
            skeleton = character_torch.skeleton
            self.model_joint_parents = skeleton.joint_parents.tolist()
            self.model_joint_names = skeleton.joint_names

            # Default export skeleton is model skeleton
            self.joint_parents = self.model_joint_parents[:]
            self.joint_names = self.model_joint_names[:]

            print(f"Found {len(self.model_joint_names)} joints in model.")
        except Exception as e:
            raise RuntimeError(f"Failed to extract hierarchy from model: {e}")

        # Extract rest pose to calculate offsets and rest quaternions
        try:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

            # Run model with zeros to get rest pose
            shape_params = torch.zeros(1, 45, device=device)
            model_params = torch.zeros(1, 204, device=device)
            expr_params = torch.zeros(1, 72, device=device)

            with torch.no_grad():
                out = model(shape_params, model_params, expr_params)
                skel_state = out[1] # [1, 127, 8]

            rest_pose_pos = skel_state[0, :, :3].cpu().numpy()
            rest_pose_quats = skel_state[0, :, 3:7].cpu().numpy()

            norms = np.linalg.norm(rest_pose_quats, axis=1)
            rest_pose_quats = rest_pose_quats / norms[:, None]

            # Store rest pose quaternions
            self.model_rest_quats = rest_pose_quats

            # Calculate offsets for model skeleton
            self.rest_offsets = []
            for i, p in enumerate(self.model_joint_parents):
                if p == -1:
                    offset = rest_pose_pos[i]
                else:
                    global_offset = rest_pose_pos[i] - rest_pose_pos[p]
                    p_quat = rest_pose_quats[p]
                    r = transform.Rotation.from_quat(p_quat)
                    r_inv = r.inv()
                    offset = r_inv.apply(global_offset)

                self.rest_offsets.append(offset)

        except Exception as e:
            raise RuntimeError(f"Failed to extract rest pose: {e}")

    def _load_target_skeleton(self, path):
        print(f"Loading target skeleton from {path}...")
        try:
            names, parents, offsets = self._read_bvh(path)
            self.joint_names = names
            self.joint_parents = parents
            self.rest_offsets = offsets

            print(f"Successfully loaded {len(names)} joints from target BVH.")
        except Exception as e:
            print(f"Error loading target skeleton: {e}")
            print("Reverting to model skeleton.")
            self.joint_names = self.model_joint_names[:]
            self.joint_parents = self.model_joint_parents[:]

    def _read_bvh(self, path):
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]

        joint_names = []
        joint_parents = []
        offsets = []

        parent_stack = [-1]

        iterator = iter(lines)
        for line in iterator:
            if line.startswith("ROOT") or line.startswith("JOINT"):
                name = line.split()[1]
                parent = parent_stack[-1]

                joint_names.append(name)
                joint_parents.append(parent)
                offsets.append(np.zeros(3))

                current_joint_idx = len(joint_names) - 1
                parent_stack.append(current_joint_idx)

            elif line.startswith("OFFSET"):
                parts = line.split()
                off = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

                if parent_stack[-1] == -2:
                    pass
                else:
                    idx = parent_stack[-1]
                    if idx >= 0 and idx < len(offsets):
                        offsets[idx] = off

            elif line.startswith("End Site"):
                parent_stack.append(-2)

            elif line.startswith("}"):
                if parent_stack:
                    parent_stack.pop()

        return joint_names, joint_parents, offsets

    def _map_rotations(self, input_quats):
        """
        Map rotations from model skeleton to target skeleton, removing MHR rest pose.

        input_quats: (F, J_in, 4) - global rotations from model (includes MHR rest pose)
        returns: (F, J_out, 4) - global rotations for target skeleton (rest pose removed)
        """
        num_frames = input_quats.shape[0]
        num_out_joints = len(self.joint_names)
        output_quats = np.zeros((num_frames, num_out_joints, 4))
        output_quats[:, :, 3] = 1.0

        mapped_count = 0

        for i, name in enumerate(self.joint_names):
            input_idx = -1

            # 1. Direct name match
            if name in self.model_joint_names:
                input_idx = self.model_joint_names.index(name)

            # 2. Dict match (MHR -> Mixamo)
            if input_idx == -1:
                for mhr_name, mix_name in self.MHR_TO_MIXAMO.items():
                    if mix_name == name:
                        if mhr_name in self.model_joint_names:
                            input_idx = self.model_joint_names.index(mhr_name)
                            break

            if input_idx != -1:
                # Get MHR rest pose for this joint
                rest_quat = self.model_rest_quats[input_idx]
                r_rest = transform.Rotation.from_quat(rest_quat)
                r_rest_inv = r_rest.inv()

                # For each frame, remove the rest pose: R_corrected = R_anim * R_rest^-1
                for frame in range(num_frames):
                    r_anim = transform.Rotation.from_quat(input_quats[frame, input_idx])
                    r_corrected = r_anim * r_rest_inv
                    output_quats[frame, i] = r_corrected.as_quat()

                mapped_count += 1

        print(f"Mapped {mapped_count}/{num_out_joints} joints with rest pose correction.")
        return output_quats

    def export(self, pred_global_rots, pred_root_pos, output_path, frame_time=0.033333):
        """
        Export motion to BVH.
        """
        # Ensure inputs are numpy
        if isinstance(pred_global_rots, torch.Tensor):
            pred_global_rots = pred_global_rots.detach().cpu().numpy()
        if isinstance(pred_root_pos, torch.Tensor):
            pred_root_pos = pred_root_pos.detach().cpu().numpy()

        # Handle single frame
        if pred_global_rots.ndim == 3 and pred_global_rots.shape[-1] == 3: # (J, 3, 3)
             pred_global_rots = pred_global_rots[None]
        if pred_global_rots.ndim == 2 and pred_global_rots.shape[-1] == 4: # (J, 4)
             pred_global_rots = pred_global_rots[None]
        if pred_root_pos.ndim == 1:
             pred_root_pos = pred_root_pos[None]

        num_frames = pred_global_rots.shape[0]
        num_joints_in = pred_global_rots.shape[1]

        if num_joints_in != len(self.model_joint_names):
            print(f"Warning: Input motion has {num_joints_in} joints, model has {len(self.model_joint_names)}")

        # Convert rotmats to quats if necessary
        if pred_global_rots.shape[-1] == 3 and pred_global_rots.shape[-2] == 3:
            flat_rots = pred_global_rots.reshape(-1, 3, 3)
            r = transform.Rotation.from_matrix(flat_rots)
            motion_quats = r.as_quat().reshape(num_frames, num_joints_in, 4)
        elif pred_global_rots.shape[-1] == 4:
             motion_quats = pred_global_rots
        else:
            raise ValueError("pred_global_rots must be (F, J, 3, 3) or (F, J, 4)")

        # Map rotations if target skeleton is different
        if self.joint_names != self.model_joint_names:
            motion_quats = self._map_rotations(motion_quats)

        with open(output_path, 'w') as f:
            self._write_hierarchy(f)
            self._write_motion(f, motion_quats, pred_root_pos, frame_time)

    def _write_hierarchy(self, f):
        f.write("HIERARCHY\n")

        # Build tree
        children = {i: [] for i in range(len(self.joint_names))}
        root_idx = -1
        for i, p in enumerate(self.joint_parents):
            if p == -1:
                root_idx = i
            else:
                children[p].append(i)

        if root_idx == -1:
             raise RuntimeError("No root found")

        self._write_joint(f, root_idx, children, 0)

    def _write_joint(self, f, idx, children, indent_level):
        indent = "  " * indent_level
        name = self.joint_names[idx]
        offset = self.rest_offsets[idx]

        parent = self.joint_parents[idx]

        if parent == -1:
            f.write(f"{indent}ROOT {name}\n")
        else:
            f.write(f"{indent}JOINT {name}\n")

        f.write(f"{indent}{{\n")
        f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

        if parent == -1:
            f.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        else:
            f.write(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n")

        if idx in children and len(children[idx]) > 0:
            for child in children[idx]:
                self._write_joint(f, child, children, indent_level + 1)
        else:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET 0.000000 0.000000 0.000000\n")
            f.write(f"{indent}  }}\n")

        f.write(f"{indent}}}\n")

    def _write_motion(self, f, motion_quats, root_pos, frame_time):
        f.write("MOTION\n")
        num_frames = motion_quats.shape[0]
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")

        for i in range(num_frames):
            row_data = []

            # Root Position
            pos = root_pos[i]
            row_data.extend([pos[0], pos[1], pos[2]])

            for j in range(len(self.joint_names)):
                # Calculate Local Rotation
                q_curr = motion_quats[i, j]

                p = self.joint_parents[j]
                if p == -1:
                    # Root local = Root global
                    q_local = q_curr
                else:
                    # q_global = q_parent * q_local
                    # q_local = q_parent^-1 * q_global
                    q_parent = motion_quats[i, p]

                    rp = transform.Rotation.from_quat(q_parent)
                    rc = transform.Rotation.from_quat(q_curr)

                    # R_local = R_parent^T * R_curr
                    rl = rp.inv() * rc
                    q_local = rl.as_quat()

                # Convert to Euler ZXY
                r_local = transform.Rotation.from_quat(q_local)
                euler = r_local.as_euler('ZXY', degrees=True)

                row_data.extend([euler[0], euler[1], euler[2]])

            f.write(" ".join([f"{x:.6f}" for x in row_data]) + "\n")
