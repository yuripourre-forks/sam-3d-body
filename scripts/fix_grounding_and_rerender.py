#!/usr/bin/env python3
"""Migrate cached pose.json files to the corrected grounding convention and
re-render mesh_render.png/mesh.ply from them, without re-running SAM3D-Body.

Background: an earlier version of apply_ground_translation() shifted both
pred_vertices AND pred_cam_t by the ground offset. Renderer.vertices_to_trimesh
places the mesh via a single `vertices + cam_t` addition, so shifting both operands
silently doubled the vertical offset applied at render time, and pred_keypoints_3d
was never shifted at all (so BVH root positions weren't actually grounded). This
script repairs cached pose.json in place:
  - pred_vertices is left as-is (it was already correctly shifted once).
  - pred_cam_t has the ground_offset subtracted back out (undoing the double-shift).
  - pred_keypoints_3d has the ground_offset added (applying the missing shift).
mesh_render.png and mesh.ply are then re-rendered from the corrected values, at the
frame's native bbox aspect ratio instead of a fixed square, so reassembly against
the original sprite sheet is pixel-accurate. mesh_iso.png is untouched: it never
used cam_t and already reflects the correct pose.
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

from sam_3d_body.visualization.renderer import Renderer  # noqa: E402
from scripts.foot_grounding import (  # noqa: E402
    compute_frame_content_bbox_and_centroid,
    render_mesh_frontal_placed,
    save_rgba_image,
    scale_content_target,
)

GROUNDING_FIX_MARKER = "cam_t_double_shift_fixed"
UPSCALE_FACTOR = 4


def save_mesh_ply(renderer: Renderer, vertices: np.ndarray, cam_t: np.ndarray, output_path: Path) -> None:
    mesh = renderer.vertices_to_trimesh(vertices, cam_t, (0.65, 0.74, 0.86))
    mesh.export(str(output_path))


def fix_and_rerender_frame(frame_dir: Path) -> bool:
    pose_path = frame_dir / "pose.json"
    mesh_path = frame_dir / "mesh.ply"
    if not pose_path.exists() or not mesh_path.exists():
        return False

    with open(pose_path, encoding="utf-8") as pose_file:
        pose_data = json.load(pose_file)

    if pose_data.get(GROUNDING_FIX_MARKER):
        return False

    ground_offset = float(pose_data["ground_offset"])
    translation = np.array([0.0, ground_offset, 0.0], dtype=np.float32)

    vertices = np.array(pose_data["pred_vertices"], dtype=np.float32)
    old_cam_t = np.array(pose_data["pred_cam_t"], dtype=np.float32)
    old_keypoints_3d = np.array(pose_data["pred_keypoints_3d"], dtype=np.float32)

    fixed_cam_t = old_cam_t - translation
    fixed_keypoints_3d = old_keypoints_3d + translation

    pose_data["pred_cam_t"] = fixed_cam_t.tolist()
    pose_data["pred_keypoints_3d"] = fixed_keypoints_3d.tolist()
    pose_data[GROUNDING_FIX_MARKER] = True

    with open(pose_path, "w", encoding="utf-8") as pose_file:
        json.dump(pose_data, pose_file, indent=2)

    faces = trimesh.load(str(mesh_path), process=False).faces
    focal_length = float(pose_data["focal_length"])
    bbox = pose_data["bbox"]
    bbox_width = int(bbox[2] - bbox[0])
    bbox_height = int(bbox[3] - bbox[1])
    keypoints_2d = np.array(pose_data["pred_keypoints_2d"], dtype=np.float32)

    renderer = Renderer(focal_length=focal_length, faces=faces)
    save_mesh_ply(renderer, vertices, fixed_cam_t, mesh_path)

    content = compute_frame_content_bbox_and_centroid(pose_data["source_frame"])
    target = scale_content_target(content, UPSCALE_FACTOR) if content is not None else None

    mesh_rgba = render_mesh_frontal_placed(
        vertices,
        faces,
        bbox_width,
        bbox_height,
        target=target,
        keypoints_2d=keypoints_2d,
    )
    save_rgba_image(mesh_rgba, str(frame_dir / "mesh_render.png"))
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix cached pose.json grounding and re-render mesh_render.png/mesh.ply"
    )
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
        fixed_count = 0
        for frame_dir in frame_dirs:
            if fix_and_rerender_frame(frame_dir):
                fixed_count += 1
        print(f"{char_dir.name}: fixed {fixed_count}/{len(frame_dirs)} frames")
    print("Done.")


if __name__ == "__main__":
    main()
