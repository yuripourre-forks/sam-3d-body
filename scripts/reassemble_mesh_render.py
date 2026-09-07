#!/usr/bin/env python3
"""Reassemble mesh_render.png frames into per-character sheets and a full sheet
matching the original sprite sheet's own layout.

mesh_render.png is rendered against the normalized square split frame (see
scripts/sprite_grid.normalize_to_square): every frame's canvas is exactly
`output_frame_size * render_scale` pixels, regardless of that frame's own
original (un-normalized, generally non-square) footprint in the sheet. To drop
each frame back into its own slot in the full, original-sheet-layout output,
this script inverts that normalization via
scripts.sprite_grid.place_normalized_render_on_sheet() -- cropping out the
content-only region of the square render, resizing it back to the frame's own
original footprint (`source_bbox` size), and pasting it at that frame's true
full-sheet position (`grid_cell_bbox` origin + `source_bbox` local offset).

Characters/sequences have different frame counts (short animations use fewer
of the sheet's frame columns than long ones), matching the original sheet
where those same columns are simply blank past a character's last real frame.
This script only ever places the frames a character actually has -- for the
full sheet that means columns past a short character's last frame are left as
background, and for the per-character sheets each strip is exactly as wide as
that character's own frame_count (never padded to match the longest
character).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.manifest_utils import inference_frame_dir_index  # noqa: E402
from scripts.sprite_grid import denormalize_render, place_normalized_render_on_sheet  # noqa: E402

DEFAULT_RENDER_SCALE = 4
BACKGROUND_RGBA = (255, 255, 255, 255)


def load_manifest(split_dir: Path) -> dict:
    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def mesh_render_path(inference_dir: Path, char_id: int, frame_idx: int) -> Path:
    return inference_dir / f"char_{char_id:02d}" / f"frame_{frame_idx:03d}" / "mesh_render.png"


def build_character_sheet(
    inference_dir: Path,
    char_entry: dict,
    output_path: Path,
    output_frame_size: int,
    render_scale: int,
) -> None:
    frame_images = []
    for frame_info in char_entry["frames"]:
        frame_path = mesh_render_path(inference_dir, char_entry["id"], inference_frame_dir_index(frame_info))
        if not frame_path.exists():
            continue
        square_rgba = np.array(Image.open(frame_path).convert("RGBA"))
        denormalized = denormalize_render(square_rgba, frame_info, output_frame_size, render_scale)
        frame_images.append(denormalized if denormalized is not None else square_rgba)

    if not frame_images:
        return

    heights = [img.shape[0] for img in frame_images]
    widths = [img.shape[1] for img in frame_images]
    sheet_height = max(heights)
    sheet_width = sum(widths)

    sheet = Image.new("RGBA", (sheet_width, sheet_height), BACKGROUND_RGBA)
    x_offset = 0
    for img in frame_images:
        frame_image = Image.fromarray(img)
        y_offset = (sheet_height - frame_image.height) // 2
        sheet.alpha_composite(frame_image, (x_offset, y_offset))
        x_offset += frame_image.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_full_sheet(
    inference_dir: Path,
    manifest: dict,
    output_path: Path,
    output_frame_size: int,
    render_scale: int,
) -> Image.Image:
    source_width, source_height = manifest["image_size"]
    sheet_width = source_width * render_scale
    sheet_height = source_height * render_scale
    sheet = Image.new("RGBA", (sheet_width, sheet_height), BACKGROUND_RGBA)

    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        for frame_info in char_entry["frames"]:
            frame_path = mesh_render_path(inference_dir, char_id, inference_frame_dir_index(frame_info))
            if not frame_path.exists():
                continue
            square_rgba = np.array(Image.open(frame_path).convert("RGBA"))
            placement = place_normalized_render_on_sheet(
                square_rgba, frame_info, output_frame_size, render_scale
            )
            if placement is None:
                continue
            resized_rgba, paste_x, paste_y = placement
            sheet.alpha_composite(Image.fromarray(resized_rgba), (paste_x, paste_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return sheet


def main():
    parser = argparse.ArgumentParser(
        description="Reassemble mesh_render.png frames into character/full sheets"
    )
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--inference-dir", default="output/inference")
    parser.add_argument("--sprites-dir", default="output/sprites_render")
    parser.add_argument("--full-output", default="output/reassembled/mesh_render_full.png")
    parser.add_argument("--render-scale", type=int, default=DEFAULT_RENDER_SCALE)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir

    inference_dir = Path(args.inference_dir)
    if not inference_dir.is_absolute():
        inference_dir = REPO_ROOT / inference_dir

    sprites_dir = Path(args.sprites_dir)
    if not sprites_dir.is_absolute():
        sprites_dir = REPO_ROOT / sprites_dir

    full_output = Path(args.full_output)
    if not full_output.is_absolute():
        full_output = REPO_ROOT / full_output

    manifest = load_manifest(split_dir)
    output_frame_size = manifest.get("output_frame_size", manifest.get("frame_size", 96))
    render_scale = args.render_scale

    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        output_path = sprites_dir / f"char_{char_id:02d}_sheet.png"
        build_character_sheet(inference_dir, char_entry, output_path, output_frame_size, render_scale)
        print(f"char_{char_id:02d}: wrote {output_path} ({char_entry['frame_count']} frames)")

    sheet = build_full_sheet(inference_dir, manifest, full_output, output_frame_size, render_scale)
    print(f"Wrote full reassembled sheet: {full_output} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
