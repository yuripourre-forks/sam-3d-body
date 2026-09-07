#!/usr/bin/env python3
"""Re-render isometric mesh images and sprite sheets from existing inference output."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam_3d_body import load_sam_3d_body
from scripts.foot_grounding import assemble_sprite_sheet, render_mesh_isometric, save_rgba_image

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/model.ckpt"
DEFAULT_MHR_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
DEFAULT_RENDER_SIZE = 512


def rerender_inference_output(
    inference_dir: Path,
    render_size: int,
    char_start: int,
    char_end: int,
) -> None:
    model, _ = load_sam_3d_body(
        checkpoint_path=str(DEFAULT_CHECKPOINT),
        mhr_path=str(DEFAULT_MHR_PATH),
        device="cuda",
    )
    faces = model.head_pose.faces.cpu().numpy()

    for char_dir in sorted(inference_dir.glob("char_*")):
        char_id = int(char_dir.name.split("_")[1])
        if char_id < char_start or char_id >= char_end:
            continue

        iso_renders = []
        for frame_dir in sorted(char_dir.glob("frame_*")):
            pose_path = frame_dir / "pose.json"
            if not pose_path.exists():
                continue
            with open(pose_path, encoding="utf-8") as pose_file:
                pose = json.load(pose_file)

            vertices = np.array(pose["pred_vertices"], dtype=np.float32)
            iso_rgba = render_mesh_isometric(vertices, faces, render_size=render_size)
            save_rgba_image(iso_rgba, str(frame_dir / "mesh_iso.png"))
            iso_renders.append(iso_rgba)

        sprites_dir = inference_dir.parent / "sprites"
        sprites_dir.mkdir(parents=True, exist_ok=True)
        if iso_renders:
            assemble_sprite_sheet(
                iso_renders,
                str(sprites_dir / f"char_{char_id:02d}_sheet.png"),
            )
        print(f"char_{char_id:02d}: re-rendered {len(iso_renders)} frames")


def main():
    parser = argparse.ArgumentParser(description="Re-render isometric mesh outputs")
    parser.add_argument("--inference-dir", default="output/inference")
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE)
    parser.add_argument("--char-start", type=int, default=0)
    parser.add_argument("--char-end", type=int, default=9)
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    if not inference_dir.is_absolute():
        inference_dir = REPO_ROOT / inference_dir

    rerender_inference_output(
        inference_dir,
        args.render_size,
        args.char_start,
        args.char_end,
    )


if __name__ == "__main__":
    main()
