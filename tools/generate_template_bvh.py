
import torch
import os
import sys

def generate_bvh(model_path, output_path):
    print(f"Loading model from {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Extract hierarchy
    try:
        character_torch = model.character_torch
        skeleton = character_torch.skeleton
        joint_parents = skeleton.joint_parents.tolist()
        joint_names = skeleton.joint_names
        
        print(f"Found {len(joint_names)} joints and {len(joint_parents)} parent indices.")
        
    except Exception as e:
        print(f"Failed to extract hierarchy: {e}")
        return

    # Extract rest pose
    try:
        shape_params = torch.zeros(1, 45)
        # Scale params are part of model_params in MHRHead logic
        # For rest pose, we want default scale. 
        # MHRHead uses scale_mean. We don't have it here easily.
        # But MHR model itself might handle zeros as "mean" if normalized?
        # Or maybe model_params being zero implies zero scale offset?
        # Let's stick with zeros.
        model_params = torch.zeros(1, 204)
        expr_params = torch.zeros(1, 72)
        
        out = model(shape_params, model_params, expr_params)
        skel_state = out[1] # [1, 127, 8]
        
        # positions are first 3 channels
        rest_pose_pos = skel_state[0, :, :3].detach().numpy() # [127, 3]
        
    except Exception as e:
        print(f"Failed to extract rest pose: {e}")
        return

    # Build tree
    children = {i: [] for i in range(len(joint_names))}
    root_idx = -1
    for i, p in enumerate(joint_parents):
        if p == -1:
            root_idx = i
        else:
            children[p].append(i)

    if root_idx == -1:
        print("Error: No root found in hierarchy.")
        return

    # Recursive function to write BVH
    def write_joint(idx, indent_level, file):
        indent = "  " * indent_level
        name = joint_names[idx]
        
        # Compute offset
        parent = joint_parents[idx]
        if parent == -1:
            offset = rest_pose_pos[idx]
        else:
            offset = rest_pose_pos[idx] - rest_pose_pos[parent]
            
        # Determine if Root or Joint
        if parent == -1:
            file.write(f"{indent}ROOT {name}\n")
        else:
            file.write(f"{indent}JOINT {name}\n")
            
        file.write(f"{indent}{{\n")
        file.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        if parent == -1:
             file.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        else:
             file.write(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
             
        # Recursively write children
        if idx in children and len(children[idx]) > 0:
            for child_idx in children[idx]:
                write_joint(child_idx, indent_level + 1, file)
        else:
            # End Site
            file.write(f"{indent}  End Site\n")
            file.write(f"{indent}  {{\n")
            # Dummy offset, or maybe estimate if we knew bone length?
            # Since we don't have child positions, use 0 0 0
            file.write(f"{indent}    OFFSET 0.000000 0.000000 0.000000\n")
            file.write(f"{indent}  }}\n")
            
        file.write(f"{indent}}}\n")

    # Write to file
    with open(output_path, 'w') as f:
        f.write("HIERARCHY\n")
        write_joint(root_idx, 0, f)
        
        f.write("MOTION\n")
        f.write("Frames: 1\n")
        f.write("Frame Time: 0.033333\n") # 30fps
        
        # Write one frame of zeros (rest pose)
        # Root has 6 channels, others 3. Total channels?
        # We need to count channels to write correct number of zeros.
        total_channels = 6 + (len(joint_names) - 1) * 3
        f.write(" ".join(["0.000000"] * total_channels) + "\n")

    print(f"Successfully wrote BVH to {output_path}")

if __name__ == "__main__":
    model_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    output_path = "assets/model_skeleton.bvh"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
    else:
        generate_bvh(model_path, output_path)

