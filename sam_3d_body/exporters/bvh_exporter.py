
import os
import torch
import numpy as np
import scipy.spatial.transform as transform

class BVHExporter:
    def __init__(self, model_path=None, model_instance=None):
        """
        Initialize the BVH exporter by loading the MHR model and extracting the skeleton.
        
        Args:
            model_path: Path to the mhr_model.pt file.
            model_instance: Pre-loaded MHR model instance (torch.jit.ScriptModule).
                            If provided, model_path is ignored.
        """
        self.model_path = model_path
        self.model = model_instance
        self.joint_names = []
        self.joint_parents = []
        self.rest_offsets = [] # Offsets in parent frame
        self.rest_global_quats = [] # For reference/debugging, or if needed
        
        self._load_and_extract_skeleton()

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

        # Extract hierarchy
        try:
            character_torch = model.character_torch
            skeleton = character_torch.skeleton
            self.joint_parents = skeleton.joint_parents.tolist()
            self.joint_names = skeleton.joint_names
            
            print(f"Found {len(self.joint_names)} joints.")
        except Exception as e:
            raise RuntimeError(f"Failed to extract hierarchy from model: {e}")

        # Extract rest pose to calculate offsets
        try:
            # Detect device where model is located
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
            
            # positions [127, 3] and quats [127, 4]
            rest_pose_pos = skel_state[0, :, :3].cpu().numpy()
            rest_pose_quats = skel_state[0, :, 3:7].cpu().numpy() # [x, y, z, w] usually for MHR/Roma
            
            # Convert quats to scipy format [x, y, z, w]
            # MHR uses [x, y, z, w]. Scipy uses [x, y, z, w].
            # Need to verify normalization
            norms = np.linalg.norm(rest_pose_quats, axis=1)
            rest_pose_quats = rest_pose_quats / norms[:, None]
            
            self.rest_global_quats = rest_pose_quats
            
            # Calculate offsets in PARENT frame
            self.rest_offsets = []
            for i, p in enumerate(self.joint_parents):
                if p == -1:
                    # Root offset is just global position (relative to world origin)
                    offset = rest_pose_pos[i]
                else:
                    # Global offset vector
                    global_offset = rest_pose_pos[i] - rest_pose_pos[p]
                    
                    # Rotate into parent frame: v_local = Q_parent^-1 * v_global
                    # q_inv = [-x, -y, -z, w]
                    p_quat = rest_pose_quats[p]
                    r = transform.Rotation.from_quat(p_quat)
                    r_inv = r.inv()
                    
                    offset = r_inv.apply(global_offset)
                
                self.rest_offsets.append(offset)
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract rest pose: {e}")

    def export(self, pred_global_rots, pred_root_pos, output_path, frame_time=0.033333):
        """
        Export motion to BVH.
        
        Args:
            pred_global_rots: (F, J, 3, 3) rotation matrices or (F, J, 4) quaternions
            pred_root_pos: (F, 3) root positions (global)
            output_path: File path to write
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
        num_joints = pred_global_rots.shape[1]
        
        if num_joints != len(self.joint_names):
            print(f"Warning: Motion has {num_joints} joints, skeleton has {len(self.joint_names)}")
        
        # Convert rotmats to quats if necessary
        if pred_global_rots.shape[-1] == 3 and pred_global_rots.shape[-2] == 3:
            # Flatten to (F*J, 3, 3) for scipy
            flat_rots = pred_global_rots.reshape(-1, 3, 3)
            r = transform.Rotation.from_matrix(flat_rots)
            # Get quats (F, J, 4)
            motion_quats = r.as_quat().reshape(num_frames, num_joints, 4)
        elif pred_global_rots.shape[-1] == 4:
             motion_quats = pred_global_rots
        else:
            raise ValueError("pred_global_rots must be (F, J, 3, 3) or (F, J, 4)")

        with open(output_path, 'w') as f:
            self._write_hierarchy(f)
            self._write_motion(f, motion_quats, pred_root_pos, frame_time)

    def _write_hierarchy(self, f):
        f.write("HIERARCHY\n")
        
        # Build tree for traversal
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
