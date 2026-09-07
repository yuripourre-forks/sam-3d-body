#!/usr/bin/env python3
"""Re-render mesh_render.png for all cached frames using render_mesh_frontal_placed.

Reuses the grounded pred_vertices/pred_keypoints_2d/bbox already cached in
pose.json (and faces from mesh.ply), so it doesn't need to re-run SAM3D-Body
inference. Run this after fix_grounding_and_rerender.py (which repairs cam_t and
pred_keypoints_3d) or after any change to render_mesh_frontal_placed itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.foot_grounding import (  # noqa: E402
    compute_frame_content_bbox_and_centroid,
    render_mesh_frontal_placed,
    save_rgba_image,
    scale_content_target,
)

UPSCALE_FACTOR = 4


def rerender_frame(frame_dir: Path) -> bool:
    pose_path = frame_dir / "pose.json"
    mesh_path = frame_dir / "mesh.ply"
    if not pose_path.exists() or not mesh_path.exists():
        return False

    with open(pose_path, encoding="utf-8") as pose_file:
        pose_data = json.load(pose_file)

    vertices = np.array(pose_data["pred_vertices"], dtype=np.float32)
    keypoints_2d = np.array(pose_data["pred_keypoints_2d"], dtype=np.float32)
    bbox = pose_data["bbox"]
    canvas_width = int(bbox[2] - bbox[0])
    canvas_height = int(bbox[3] - bbox[1])
    faces = trimesh.load(str(mesh_path), process=False).faces

    content = compute_frame_content_bbox_and_centroid(pose_data["source_frame"])
    target = scale_content_target(content, UPSCALE_FACTOR) if content is not None else None

    mesh_rgba = render_mesh_frontal_placed(
        vertices,
        faces,
        canvas_width,
        canvas_height,
        target=target,
        keypoints_2d=keypoints_2d,
    )
    save_rgba_image(mesh_rgba, str(frame_dir / "mesh_render.png"))
    return True


def main():
    parser = argparse.ArgumentParser(description="Re-render mesh_render.png from cached pose data")
    parser.add_argument("--inference-dir", default="output/inference")
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    if not inference_dir.is_absolute():
        inference_dir = REPO_ROOT / inference_dir

    char_dirs = sorted(
        (char_dir for char_dir in inference_dir.iterdir() if char_dir.is_dir()),
        key=lambda char_dir: char_dir.name,
    )
    for char_dir in char_dirs:
        frame_dirs = sorted(
            (frame_dir for frame_dir in char_dir.iterdir() if frame_dir.is_dir()),
            key=lambda frame_dir: frame_dir.name,
        )
        rerendered_count = sum(rerender_frame(frame_dir) for frame_dir in frame_dirs)
        print(f"{char_dir.name}: re-rendered {rerendered_count}/{len(frame_dirs)} frames")
    print("Done.")


if __name__ == "__main__":
    main()
