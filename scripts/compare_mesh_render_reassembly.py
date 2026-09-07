#!/usr/bin/env python3
"""Compare the reassembled mesh_render sheet against the original sprite sheet.

For every frame in the split manifest, this measures how well the SAM3D-Body
mesh_render silhouette overlaps the original sprite's own silhouette. Per-frame
metrics (IoU, centroid offset, bbox-size ratio) are computed directly in the
normalized-square space both mesh_render.png and the on-disk split frame
already share (see scripts/sprite_grid.normalize_to_square) -- no coordinate
conversion needed there. The full-sheet visual overlay (original in red,
mesh_render in green, overlap in yellow), however, must place each frame's
mesh_render.png back onto the *original*, un-normalized sheet layout, which
requires inverting that per-frame normalization via
scripts.sprite_grid.place_normalized_render_on_sheet().

Generalized for both single-angle (townsfolk) and multi-angle (succubus)
sheets: content masking uses scripts.sprite_grid.content_mask (magenta +
white + alpha), the same background-detection convention already used by
scripts/split_sprite_sheet.py, rather than the townsfolk-only magenta-only
masking this script used previously. On-disk frame directories are looked up
via scripts.manifest_utils.inference_frame_dir_index (the local per-character
index), not the sheet-global "index" field, matching every other stage script.
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
from scripts.sprite_grid import content_mask as sheet_content_mask  # noqa: E402
from scripts.sprite_grid import place_normalized_render_on_sheet  # noqa: E402

DEFAULT_RENDER_SCALE = 4
OVERLAY_ORIGINAL_RGB = (230, 30, 30)
OVERLAY_MESH_RGB = (30, 200, 60)
OVERLAY_OVERLAP_RGB = (230, 210, 30)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return float(cols.mean()), float(rows.mean())


def mask_bbox_size(mask: np.ndarray) -> tuple[int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)


def compare_frame(
    split_dir: Path,
    frame_info: dict,
    mesh_render_path: Path,
) -> dict | None:
    """Compare a frame's own normalized split image against its mesh_render.png.

    Both images already live in the same coordinate space (the
    `output_frame_size`-normalized square, mesh_render.png upscaled by
    `render_scale`), so the sprite mask only needs a plain resize -- no bbox
    conversion is required here.
    """
    if not mesh_render_path.exists():
        return None
    mesh_rgba = np.array(Image.open(mesh_render_path).convert("RGBA"))
    mesh_mask = mesh_rgba[..., 3] > 0

    sprite_rgba = Image.open(split_dir / frame_info["path"]).convert("RGBA")
    sprite_upscaled = np.array(
        sprite_rgba.resize((mesh_rgba.shape[1], mesh_rgba.shape[0]), Image.NEAREST)
    )
    original_mask = sheet_content_mask(sprite_upscaled)

    intersection = int(np.logical_and(original_mask, mesh_mask).sum())
    union = int(np.logical_or(original_mask, mesh_mask).sum())
    iou = intersection / union if union > 0 else None

    original_centroid = mask_centroid(original_mask)
    mesh_centroid = mask_centroid(mesh_mask)
    centroid_offset = None
    if original_centroid is not None and mesh_centroid is not None:
        centroid_offset = float(
            np.hypot(
                original_centroid[0] - mesh_centroid[0],
                original_centroid[1] - mesh_centroid[1],
            )
        )

    original_size = mask_bbox_size(original_mask)
    mesh_size = mask_bbox_size(mesh_mask)
    size_ratio = None
    if original_size is not None and mesh_size is not None:
        original_area = original_size[0] * original_size[1]
        mesh_area = mesh_size[0] * mesh_size[1]
        if original_area > 0:
            size_ratio = mesh_area / original_area

    return {
        "iou": iou,
        "centroid_offset_px": centroid_offset,
        "bbox_area_ratio": size_ratio,
    }


def build_overlay(
    original_full_rgba: np.ndarray,
    inference_dir: Path,
    manifest: dict,
    output_frame_size: int,
    render_scale: int,
) -> Image.Image:
    sheet_width, sheet_height = original_full_rgba.shape[1], original_full_rgba.shape[0]
    overlay = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)

    original_mask_full = sheet_content_mask(original_full_rgba)

    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        for frame_info in char_entry["frames"]:
            mesh_path = (
                inference_dir
                / f"char_{char_id:02d}"
                / f"frame_{inference_frame_dir_index(frame_info):03d}"
                / "mesh_render.png"
            )
            if not mesh_path.exists():
                continue
            square_rgba = np.array(Image.open(mesh_path).convert("RGBA"))
            placement = place_normalized_render_on_sheet(
                square_rgba, frame_info, output_frame_size, render_scale
            )
            if placement is None:
                continue
            resized_rgba, paste_x, paste_y = placement
            paste_h, paste_w = resized_rgba.shape[:2]
            mesh_mask = resized_rgba[:, :, 3] > 0
            overlay[paste_y : paste_y + paste_h, paste_x : paste_x + paste_w][mesh_mask] = OVERLAY_MESH_RGB

    overlap_mask = original_mask_full & np.any(overlay != 0, axis=-1)
    overlay[original_mask_full & ~np.any(overlay != 0, axis=-1)] = OVERLAY_ORIGINAL_RGB
    overlay[overlap_mask] = OVERLAY_OVERLAP_RGB
    return Image.fromarray(overlay)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    parser = argparse.ArgumentParser(
        description="Compare reassembled mesh_render sheet against the original sprite sheet"
    )
    parser.add_argument("--original", required=True, help="Path to the original sprite sheet PNG")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--inference-dir", default="output/inference")
    parser.add_argument("--render-scale", type=int, default=DEFAULT_RENDER_SCALE)
    parser.add_argument("--overlay-output", default="output/reassembled/comparison_overlay.png")
    parser.add_argument("--report-output", default="output/reassembled/comparison_report.json")
    parser.add_argument(
        "--save-original-upscaled",
        default=None,
        help="Optional path to also save the nearest-neighbor-upscaled original sheet "
        "(same resolution as the mesh_render sheet), for direct visual comparison",
    )
    args = parser.parse_args()

    original_path = resolve_path(args.original)
    split_dir = resolve_path(args.split_dir)
    inference_dir = resolve_path(args.inference_dir)
    overlay_output = resolve_path(args.overlay_output)
    report_output = resolve_path(args.report_output)
    render_scale = args.render_scale

    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    output_frame_size = manifest.get("output_frame_size", manifest.get("frame_size", 96))

    original_image = Image.open(original_path).convert("RGBA")
    original_full_rgba = np.array(
        original_image.resize(
            (original_image.width * render_scale, original_image.height * render_scale),
            Image.NEAREST,
        )
    )

    if args.save_original_upscaled:
        original_upscaled_path = resolve_path(args.save_original_upscaled)
        original_upscaled_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(original_full_rgba).save(original_upscaled_path)
        print(f"Wrote upscaled original sheet: {original_upscaled_path}")

    per_character_report = []
    all_ious = []
    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        frame_reports = []
        for frame_info in char_entry["frames"]:
            mesh_path = (
                inference_dir
                / f"char_{char_id:02d}"
                / f"frame_{inference_frame_dir_index(frame_info):03d}"
                / "mesh_render.png"
            )
            result = compare_frame(split_dir, frame_info, mesh_path)
            if result is None:
                continue
            result["frame_index"] = frame_info["index"]
            frame_reports.append(result)
            if result["iou"] is not None:
                all_ious.append(result["iou"])

        ious = [r["iou"] for r in frame_reports if r["iou"] is not None]
        per_character_report.append(
            {
                "char_id": char_id,
                "mean_iou": float(np.mean(ious)) if ious else None,
                "min_iou": float(np.min(ious)) if ious else None,
                "frames": frame_reports,
            }
        )
        print(
            f"char_{char_id:02d}: mean IoU = {np.mean(ious):.3f}, min IoU = {np.min(ious):.3f}"
            if ious
            else f"char_{char_id:02d}: no comparable frames"
        )

    print(f"\nOverall mean IoU: {np.mean(all_ious):.3f}" if all_ious else "No comparable frames")

    report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as report_file:
        json.dump(
            {"overall_mean_iou": float(np.mean(all_ious)) if all_ious else None, "characters": per_character_report},
            report_file,
            indent=2,
        )
    print(f"Wrote report: {report_output}")

    overlay_image = build_overlay(original_full_rgba, inference_dir, manifest, output_frame_size, render_scale)
    overlay_output.parent.mkdir(parents=True, exist_ok=True)
    overlay_image.save(overlay_output)
    print(f"Wrote overlay: {overlay_output}")


if __name__ == "__main__":
    main()
