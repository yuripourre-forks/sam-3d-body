
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
        self.end_site_offsets = {} # Dict mapping joint_idx -> end site offset

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
            names, parents, offsets, end_site_offsets = self._read_bvh(path)
            self.joint_names = names
            self.joint_parents = parents
            self.rest_offsets = offsets
            self.end_site_offsets = end_site_offsets

            print(f"Successfully loaded {len(names)} joints from target BVH.")
            print(f"Found {len(end_site_offsets)} End Site offsets.")
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
        end_site_offsets = {}

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
                    # This is an End Site offset
                    # The parent of the End Site is the joint before it
                    if len(parent_stack) >= 2:
                        parent_joint_idx = parent_stack[-2]
                        if parent_joint_idx >= 0:
                            end_site_offsets[parent_joint_idx] = off
                else:
                    idx = parent_stack[-1]
                    if idx >= 0 and idx < len(offsets):
                        offsets[idx] = off

            elif line.startswith("End Site"):
                parent_stack.append(-2)

            elif line.startswith("}"):
                if parent_stack:
                    parent_stack.pop()

        return joint_names, joint_parents, offsets, end_site_offsets

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

    def _compute_special_head_offsets(self, motion_quats, root_pos):
        """
        Compute special offsets for head-related joints based on the first frame.

        This computes:
        - headfront: base at head joint base, pointing forward
        - head_end: pointing up based on spine direction
        """
        # Use first frame for computing offsets
        frame_quats = motion_quats[0]  # (J, 4)
        frame_root = root_pos[0]  # (3,)

        # Compute global positions for first frame
        global_positions = self._compute_joint_global_positions(frame_quats, frame_root)

        # Find key joint indices
        head_idx = self.joint_names.index("Head") if "Head" in self.joint_names else -1
        neck_idx = self.joint_names.index("neck") if "neck" in self.joint_names else -1
        headfront_idx = self.joint_names.index("headfront") if "headfront" in self.joint_names else -1
        head_end_idx = self.joint_names.index("head_end") if "head_end" in self.joint_names else -1

        HEAD_LENGTH = 0.2  # Approximate head length in meters
        FORWARD_OFFSET = 0.05  # How far forward from head base to position headfront (5cm)
        FORWARD_LENGTH = 0.15  # Length for headfront bone end site

        if head_idx != -1:
            # Get head and neck positions
            head_pos = global_positions[head_idx]

            # Get the Head joint's offset (this tells us where the base is relative to parent)
            head_offset = self.rest_offsets[head_idx]
            head_offset_length = np.linalg.norm(head_offset)

            # Determine head up vector and base position
            if neck_idx != -1:
                neck_pos = global_positions[neck_idx]
                # Head vector (from neck to head joint position)
                head_vector = head_pos - neck_pos
                head_length = np.linalg.norm(head_vector)
                if head_length > 0:
                    head_up = head_vector / head_length
                else:
                    head_up = np.array([0.0, 1.0, 0.0])
            else:
                # No neck, use head offset direction or default up
                if head_offset_length > 0:
                    head_up = head_offset / head_offset_length
                else:
                    head_up = np.array([0.0, 1.0, 0.0])

            # Compute forward direction based on head and neck orientation
            forward_dir = self._compute_head_forward_direction(frame_quats, global_positions)

            # Compute up direction based on spine
            up_dir = self._compute_spine_up_direction(frame_quats, global_positions)

            # For headfront: positioned at the base of the head (where it connects to parent)
            # with a small forward offset
            if headfront_idx != -1:
                # Get head rotation to convert to local frame
                head_quat = frame_quats[head_idx]
                r_head = transform.Rotation.from_quat(head_quat)
                r_head_inv = r_head.inv()

                # The base of the head joint is at offset 0 in the head's local frame
                # But we want to offset it slightly forward and down
                # Down: use negative head up direction (toward base)
                # Forward: use forward direction

                # Position headfront at the base with forward offset
                base_offset_world = -head_up * (head_offset_length * 0.5) + forward_dir * FORWARD_OFFSET

                # Convert to head's local frame
                local_offset = r_head_inv.apply(base_offset_world)

                # This is the offset for headfront joint
                self._computed_headfront_offset = local_offset

                # End site: extend forward from headfront position
                forward_world = forward_dir * FORWARD_LENGTH
                local_forward = r_head_inv.apply(forward_world)
                self._computed_headfront_end_offset = local_forward

            # For head_end: blend the original offset with spine up direction
            if head_end_idx != -1:
                # Get original offset
                original_offset = self.rest_offsets[head_end_idx]
                original_length = np.linalg.norm(original_offset)

                if original_length == 0:
                    original_length = HEAD_LENGTH

                # Get head rotation to convert to local frame
                head_quat = frame_quats[head_idx]
                r_head = transform.Rotation.from_quat(head_quat)
                r_head_inv = r_head.inv()

                # Blend head up and spine up (more weight on spine up)
                SPINE_WEIGHT = 0.7
                blended_up = (1.0 - SPINE_WEIGHT) * head_up + SPINE_WEIGHT * up_dir
                blended_up = blended_up / np.linalg.norm(blended_up)

                # Scale to original length
                new_offset_world = blended_up * original_length

                # Convert to head's local frame
                new_offset_local = r_head_inv.apply(new_offset_world)
                self._computed_head_end_offset = new_offset_local

                # End site: continue in same direction
                end_site_world = blended_up * (original_length * 0.8)
                end_site_local = r_head_inv.apply(end_site_world)
                self._computed_head_end_end_offset = end_site_local

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

        # Compute special head offsets based on first frame
        self._compute_special_head_offsets(motion_quats, pred_root_pos)

        with open(output_path, 'w') as f:
            self._write_hierarchy(f)
            self._write_motion(f, motion_quats, pred_root_pos, frame_time)

    def _compute_joint_global_positions(self, global_quats, root_pos):
        """
        Compute global positions for all joints given global rotations and root position.

        Args:
            global_quats: (J, 4) - global quaternions for all joints
            root_pos: (3,) - root position

        Returns:
            global_positions: (J, 3) - global positions for all joints
        """
        num_joints = len(self.joint_names)
        global_positions = np.zeros((num_joints, 3))

        for i in range(num_joints):
            parent_idx = self.joint_parents[i]

            if parent_idx == -1:
                # Root joint
                global_positions[i] = root_pos
            else:
                # Parent's global position and rotation
                parent_pos = global_positions[parent_idx]
                parent_quat = global_quats[parent_idx]

                # Transform offset from parent frame to world frame
                r_parent = transform.Rotation.from_quat(parent_quat)
                offset_world = r_parent.apply(self.rest_offsets[i])

                global_positions[i] = parent_pos + offset_world

        return global_positions

    def _compute_head_forward_direction(self, global_quats, global_positions):
        """
        Compute the forward direction for the head based on head and neck orientation.

        Returns:
            forward_dir: (3,) - normalized forward direction vector
        """
        # Find head and neck indices
        head_idx = self.joint_names.index("Head") if "Head" in self.joint_names else -1
        neck_idx = self.joint_names.index("neck") if "neck" in self.joint_names else -1

        if head_idx == -1:
            return np.array([0.0, 0.0, -1.0])  # Default forward

        # Get head global rotation
        head_quat = global_quats[head_idx]
        r_head = transform.Rotation.from_quat(head_quat)

        if neck_idx != -1:
            # Blend head and neck rotations for more stable forward direction
            neck_quat = global_quats[neck_idx]
            r_neck = transform.Rotation.from_quat(neck_quat)

            # Average the two rotations using SLERP (50/50 blend)
            from scipy.spatial.transform import Slerp
            key_times = [0, 1]
            key_rots = transform.Rotation.from_quat([neck_quat, head_quat])
            slerp = Slerp(key_times, key_rots)
            r_avg = slerp(0.5)
        else:
            r_avg = r_head

        # Forward is negative Z in the local frame (BVH convention)
        local_forward = np.array([0.0, 0.0, -1.0])
        forward_dir = r_avg.apply(local_forward)

        return forward_dir / np.linalg.norm(forward_dir)

    def _compute_spine_up_direction(self, global_quats, global_positions):
        """
        Compute the up direction based on the spine orientation.

        Returns:
            up_dir: (3,) - normalized up direction vector
        """
        # Find spine joints
        spine_joints = []
        for name in ["Spine", "Spine01", "Spine02"]:
            if name in self.joint_names:
                spine_joints.append(self.joint_names.index(name))

        if len(spine_joints) == 0:
            return np.array([0.0, 1.0, 0.0])  # Default up

        # Use the highest spine joint we can find
        spine_idx = spine_joints[-1]
        spine_quat = global_quats[spine_idx]
        r_spine = transform.Rotation.from_quat(spine_quat)

        # Up is positive Y in the local frame
        local_up = np.array([0.0, 1.0, 0.0])
        up_dir = r_spine.apply(local_up)

        return up_dir / np.linalg.norm(up_dir)

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

        # Special handling for head-related joints
        # These offsets will be computed dynamically based on pose
        if name == "headfront":
            # headfront should start at the base of the head and point forward
            # We'll compute this based on the first frame of the animation
            if hasattr(self, '_computed_headfront_offset'):
                offset = self._computed_headfront_offset
        elif name == "head_end":
            # head_end should point up based on spine direction
            if hasattr(self, '_computed_head_end_offset'):
                offset = self._computed_head_end_offset

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
            # Write End Site with correct offset
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")

            # Special handling for headfront end site
            if name == "headfront" and hasattr(self, '_computed_headfront_end_offset'):
                end_offset = self._computed_headfront_end_offset
                f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            elif name == "head_end" and hasattr(self, '_computed_head_end_end_offset'):
                end_offset = self._computed_head_end_end_offset
                f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            elif idx in self.end_site_offsets:
                # Use End Site offset from template if available
                end_offset = self.end_site_offsets[idx]
                f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            else:
                # Fallback to zero offset if not found
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
