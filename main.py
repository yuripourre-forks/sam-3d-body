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
from sam_3d_body.exporters.bvh_exporter import BVHExporter
from tqdm import tqdm

EPSILON = 1e-6


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

    # Initialize BVH exporter using the loaded MHR model instance
    # Access the underlying MHR model from the SAM3DBody model structure
    # model -> head_pose -> mhr
    
    # Resolve rest pose path
    if args.rest_pose == "mixamo_skeleton":
        rest_pose_path = os.path.join("assets", "mixamo_skeleton.bvh")
    elif args.rest_pose == "sam_skeleton":
        rest_pose_path = os.path.join("assets", "sam_skeleton.bvh")
    else:
        rest_pose_path = args.rest_pose
        
    if rest_pose_path and not os.path.exists(rest_pose_path):
        # Check relative to project root if not found
        if os.path.exists(os.path.join(root, rest_pose_path)):
             rest_pose_path = os.path.join(root, rest_pose_path)
        else:
             print(f"Warning: Rest pose BVH not found at {rest_pose_path}. Using default model skeleton.")
             rest_pose_path = None

    try:
        mhr_instance = model.head_pose.mhr
        bvh_exporter = BVHExporter(model_instance=mhr_instance, target_skeleton_path=rest_pose_path)
    except AttributeError:
        print("Warning: Could not find loaded MHR model instance. Trying to load from path if available.")
        # Fallback to path if possible (though model.head_pose.mhr should exist)
        # We need to construct the path to mhr_model.pt
        if mhr_path:
             if os.path.isdir(mhr_path):
                 mhr_file = os.path.join(mhr_path, "mhr_model.pt")
             else:
                 mhr_file = mhr_path
             bvh_exporter = BVHExporter(model_path=mhr_file, target_skeleton_path=rest_pose_path)
        else:
             raise RuntimeError("Could not initialize BVH Exporter: MHR model instance not found and mhr_path not provided.")

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
            # Get global rotations (127, 3, 3)
            joint_rotations = output.get("pred_global_rots")
            
            # Get joint coordinates to extract root position
            joint_coords = output.get("pred_joint_coords") # (127, 3)

            if joint_rotations is None or joint_coords is None:
                print(f"Missing pose data for {image_path} person {idx}, skipping BVH export.")
                continue

            # Convert to numpy if needed
            if isinstance(joint_rotations, torch.Tensor):
                joint_rotations = joint_rotations.cpu().numpy()
            if isinstance(joint_coords, torch.Tensor):
                joint_coords = joint_coords.cpu().numpy()

            # Root position is index 0
            root_pos = joint_coords[0]

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
                bvh_exporter.export(joint_rotations, root_pos, bvh_path)
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
        "--rest_pose",
        default="mixamo_skeleton",
        type=str,
        help="Rest pose skeleton to use (mixamo_skeleton, sam_skeleton, or path to BVH file)",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    args = parser.parse_args()

    main(args)
