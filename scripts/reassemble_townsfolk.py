#!/usr/bin/env python3
"""Reassemble split frames back into a sprite sheet using manifest bboxes."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MAGENTA_RGBA = (255, 0, 255, 255)


def reassemble_from_manifest(
    split_dir: Path,
    output_path: Path,
    source_size: tuple[int, int] | None = None,
    frames_dir: Path | None = None,
    scale: int = 1,
) -> Image.Image:
    """Reassemble a sprite sheet from per-frame images referenced by the manifest.

    `frames_dir` lets the actual pixel data come from a different directory than the
    one holding manifest.json (e.g. an `upscaled/` mirror of `split/`), as long as it
    has the same char_XX/frame_YYY.png layout. `scale` multiplies the manifest's
    source_bbox coordinates so an upscaled sheet can be reassembled at its native
    (larger) resolution.
    """
    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    frames_dir = frames_dir if frames_dir is not None else split_dir

    if source_size is None:
        source_width, source_height = manifest["image_size"]
    else:
        source_width, source_height = source_size
    source_width *= scale
    source_height *= scale

    sheet = Image.new("RGBA", (source_width, source_height), MAGENTA_RGBA)

    for char_entry in manifest["characters"]:
        for frame_info in char_entry["frames"]:
            frame_path = frames_dir / frame_info["path"]
            if not frame_path.exists():
                continue
            frame_image = Image.open(frame_path).convert("RGBA")
            x0, y0, x1, y1 = frame_info["source_bbox"]
            x0, y0, x1, y1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
            bbox_width, bbox_height = x1 - x0, y1 - y0
            if frame_image.size != (bbox_width, bbox_height):
                if scale == 1:
                    frame_image = frame_image.crop((0, 0, bbox_width, bbox_height))
                else:
                    frame_image = frame_image.resize((bbox_width, bbox_height))
            sheet.paste(frame_image, (x0, y0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return sheet


WHITE_COMPARE_THRESHOLD = 240


def compare_with_source(
    reassembled: Image.Image,
    source_path: Path,
) -> dict:
    """Compare pixel-for-pixel, plus a content-only ratio that ignores blank white
    canvas space (which split_townsfolk.py intentionally does not extract as frames).
    """
    source = Image.open(source_path).convert("RGBA")
    reassembled = reassembled.crop((0, 0, source.width, source.height))

    source_arr = np.array(source)
    reassembled_arr = np.array(reassembled)
    matching_pixels = np.all(source_arr == reassembled_arr, axis=2)
    match_ratio = float(matching_pixels.mean())

    r, g, b = source_arr[:, :, 0], source_arr[:, :, 1], source_arr[:, :, 2]
    is_white = (
        (r > WHITE_COMPARE_THRESHOLD)
        & (g > WHITE_COMPARE_THRESHOLD)
        & (b > WHITE_COMPARE_THRESHOLD)
    )
    content_mask = ~is_white
    content_total = int(content_mask.sum())
    content_matching = int((matching_pixels & content_mask).sum())
    content_match_ratio = content_matching / content_total if content_total else 1.0

    return {
        "source_size": [source.width, source.height],
        "match_ratio": match_ratio,
        "matching_pixels": int(matching_pixels.sum()),
        "total_pixels": int(matching_pixels.size),
        "content_match_ratio": content_match_ratio,
        "content_matching_pixels": content_matching,
        "content_total_pixels": content_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Reassemble townsfolk sprite sheet")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--output", default="output/reassembled/townsfolk.png")
    parser.add_argument("--source", default="input/townsfolk.png")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Directory holding char_XX/frame_YYY.png pixel data (default: --split-dir)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Multiply manifest source_bbox coordinates by this factor "
        "(use with an upscaled --frames-dir)",
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path

    frames_dir = None
    if args.frames_dir is not None:
        frames_dir = Path(args.frames_dir)
        if not frames_dir.is_absolute():
            frames_dir = REPO_ROOT / frames_dir

    reassembled = reassemble_from_manifest(
        split_dir, output_path, frames_dir=frames_dir, scale=args.scale
    )
    print(f"Wrote {output_path} ({reassembled.size[0]}x{reassembled.size[1]})")

    if args.compare:
        stats = compare_with_source(reassembled, source_path)
        print(
            f"Full-canvas pixel match vs source: {stats['match_ratio']*100:.2f}% "
            f"({stats['matching_pixels']}/{stats['total_pixels']})"
        )
        print(
            f"Content-only pixel match vs source: {stats['content_match_ratio']*100:.3f}% "
            f"({stats['content_matching_pixels']}/{stats['content_total_pixels']})"
        )


if __name__ == "__main__":
    main()
