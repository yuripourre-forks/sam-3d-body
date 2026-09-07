#!/usr/bin/env python3
"""Re-render mesh_iso.png + sprite sheets from existing pose.json/mesh.ply outputs.

Use this after changing the isometric camera logic in foot_grounding.py: it reuses
the already-computed grounded vertices/faces cached on disk by process_townsfolk.py,
so it doesn't need to re-run SAM3D-Body inference (which is by far the slow step).
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
    assemble_sprite_sheet,
    render_mesh_isometric,
    save_rgba_image,
)

DEFAULT_RENDER_SIZE = 512


def rerender_character(char_dir: Path, render_size: int) -> None:
    frame_dirs = sorted(
        (frame_dir for frame_dir in char_dir.iterdir() if frame_dir.is_dir()),
        key=lambda frame_dir: frame_dir.name,
    )

    iso_renders = []
    for frame_dir in frame_dirs:
        pose_path = frame_dir / "pose.json"
        mesh_path = frame_dir / "mesh.ply"
        if not pose_path.exists() or not mesh_path.exists():
            print(f"  skipping {frame_dir.name}: missing pose.json/mesh.ply")
            continue

        with open(pose_path, encoding="utf-8") as pose_file:
            pose_data = json.load(pose_file)
        vertices = np.array(pose_data["pred_vertices"], dtype=np.float32)
        faces = trimesh.load(str(mesh_path), process=False).faces

        iso_rgba = render_mesh_isometric(vertices, faces, render_size=render_size)
        save_rgba_image(iso_rgba, str(frame_dir / "mesh_iso.png"))
        iso_renders.append(iso_rgba)

    return iso_renders


def main():
    parser = argparse.ArgumentParser(description="Re-render isometric sprites from cached pose data")
    parser.add_argument("--inference-dir", default="output/inference")
    parser.add_argument("--sprites-dir", default="output/sprites")
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE)
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    if not inference_dir.is_absolute():
        inference_dir = REPO_ROOT / inference_dir

    sprites_dir = Path(args.sprites_dir)
    if not sprites_dir.is_absolute():
        sprites_dir = REPO_ROOT / sprites_dir
    sprites_dir.mkdir(parents=True, exist_ok=True)

    char_dirs = sorted(
        (char_dir for char_dir in inference_dir.iterdir() if char_dir.is_dir()),
        key=lambda char_dir: char_dir.name,
    )
    for char_dir in char_dirs:
        print(f"Re-rendering {char_dir.name}...")
        iso_renders = rerender_character(char_dir, args.render_size)
        if iso_renders:
            assemble_sprite_sheet(
                iso_renders,
                str(sprites_dir / f"{char_dir.name}_sheet.png"),
            )
    print("Done.")


if __name__ == "__main__":
    main()
