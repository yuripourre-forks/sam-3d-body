
import torch
import sys
import os

def inspect_mhr(model_path):
    print(f"Loading model from {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Model loaded successfully.")
    print("Model graph:", model.graph)
    
    # Try to find constants or attributes related to hierarchy
    # Iterating over named parameters/buffers might help
    for name, param in model.named_parameters():
        print(f"Param: {name}, shape: {param.shape}")
        
    try:
        character_torch = model.character_torch
        print("Found character_torch submodule/attribute")
        # Try to list attributes of character_torch
        # We can inspect its type or dir()
        print(f"Type: {type(character_torch)}")
        print(f"Dir: {dir(character_torch)}")
        
        if hasattr(character_torch, 'parents'):
            print("Found 'parents' attribute in character_torch!")
            parents = character_torch.parents
            print(f"Parents shape: {parents.shape}")
            print(f"Parents: {parents}")
        else:
             # Check named buffers/parameters of character_torch
             for name, buf in character_torch.named_buffers():
                 if 'parent' in name:
                     print(f"Found buffer in character_torch: {name}")
                     print(buf)
        
        # Check for joint names
        skeleton = getattr(character_torch, 'skeleton', None)
        if skeleton is not None:
            print(f"Skeleton attributes: {dir(skeleton)}")
            if hasattr(skeleton, 'joint_names'):
                print(f"Found joint_names: {skeleton.joint_names}")
            else:
                print("No joint_names in skeleton object")

    except Exception as e:
        print(f"Could not access character_torch: {e}")

    # Try to run with dummy inputs to get rest pose
    # Based on MHRHead logic
    # shape_params: [1, 45]
    # model_params: [1, 204] (Assuming 136 pose + 68 scale)
    # expr_params: [1, 72]
    
    shape_params = torch.zeros(1, 45)
    model_params = torch.zeros(1, 204) 
    # Set scale params? In MHRHead, scales = scale_mean + pred * comps. 
    # If we want rest pose, maybe zero is fine, or maybe we need scale_mean?
    # But we don't have scale_mean here easily without MHRHead.
    # Let's try zeros first.
    
    expr_params = torch.zeros(1, 72)
    
    try:
        # We might need to adjust input shapes if my inference was wrong
        # Let's try to run
        out = model(shape_params, model_params, expr_params)
        print("Inference successful with zeros.")
        if isinstance(out, tuple):
            print(f"Output tuple len: {len(out)}")
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    print(f"Output {i} shape: {o.shape}")
                else:
                    print(f"Output {i} type: {type(o)}")
            
            # Usually out[1] is curr_skel_state
            # curr_skel_state: [B, 127, 8] ?
            # MHRHead: curr_joint_coords, curr_joint_quats, _ = torch.split(curr_skel_state, [3, 4, 1], dim=2)
            
            if len(out) >= 2:
                skel_state = out[1]
                print(f"Skel state shape: {skel_state.shape}")
                
                coords = skel_state[0, :, :3]
                print("Joint coords (first 5):")
                print(coords[:5])

                rest_pose_quats = skel_state[0, :, 3:7]
                print(f"Rest pose quats shape: {rest_pose_quats.shape}")
                print("Sample quats (first 5):")
                print(rest_pose_quats[:5])
                
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    model_path = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
    else:
        inspect_mhr(model_path)

