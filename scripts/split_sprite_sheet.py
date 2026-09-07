#!/usr/bin/env python3
"""Generalized sprite-sheet splitter for fixed-camera animation sheets.

Supports:
- townsfolk-style: one row per character, single view angle, variable row height
- succubus-style: fixed-pitch cells, 8 view angles per sequence, white grid lines
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

from scripts.sprite_config import SheetConfig, load_sheet_config
from scripts.sprite_grid import (
    DEFAULT_WHITE_THRESHOLD,
    content_regions_between_separators,
    count_used_cells,
    draw_grid_preview,
    extract_cell,
    find_separator_bands,
    normalize_to_square,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOWNSFOLK_NUM_CHARACTERS = 9
TOWNSFOLK_FRAME_SIZE = 96


def detect_townsfolk_layout(
    rgba: np.ndarray,
    num_characters: int,
    frame_size: int,
) -> list[tuple[int, int]]:
    from scripts.split_townsfolk import detect_row_layout

    return detect_row_layout(rgba, num_characters, frame_size)


def detect_multi_angle_blocks(
    rgba: np.ndarray,
    config: SheetConfig,
) -> list[dict]:
    if config.blocks:
        return _blocks_from_config(config.blocks, config)

    white_threshold = config.white_threshold
    white_fraction_threshold = config.white_fraction_threshold
    rgb = rgba[..., :3].astype(np.int32)
    white = (rgb[..., 0] > white_threshold) & (rgb[..., 1] > white_threshold) & (rgb[..., 2] > white_threshold)

    row_fraction = white.mean(axis=1)
    row_separators = find_separator_bands(row_fraction, white_fraction_threshold, min_band_size=1)
    sequence_regions = content_regions_between_separators(rgba.shape[0], row_separators, min_region_size=32)

    blocks: list[dict] = []
    sequence_names = [
        sequence.name
        for character in config.characters
        for sequence in character.sequences
    ]
    sequence_index = 0

    for region_y0, region_y1 in sequence_regions:
        region_height = region_y1 - region_y0 + 1
        if region_height < config.num_view_angles * 8:
            continue

        region_white = white[region_y0 : region_y1 + 1]
        col_fraction = region_white.mean(axis=0)
        col_separators = find_separator_bands(col_fraction, white_fraction_threshold, min_band_size=1)
        x_regions = content_regions_between_separators(rgba.shape[1], col_separators, min_region_size=64)

        stripe_height = region_height // config.num_view_angles
        remainder = region_height - stripe_height * config.num_view_angles
        stripe_offsets = []
        cursor = region_y0
        for angle_index in range(config.num_view_angles):
            extra = 1 if angle_index < remainder else 0
            stripe_offsets.append((cursor, cursor + stripe_height + extra - 1))
            cursor += stripe_height + extra

        for x0, x1 in x_regions:
            cell_size = stripe_height
            if config.grid_cell_size is not None:
                cell_size = config.grid_cell_size
            block_width = x1 - x0 + 1
            num_cols = block_width // cell_size
            if num_cols <= 0:
                continue

            sequence_name = (
                sequence_names[sequence_index]
                if sequence_index < len(sequence_names)
                else f"sequence_{sequence_index:02d}"
            )
            sequence_index += 1

            stripes = []
            for angle_index, (stripe_y0, stripe_y1) in enumerate(stripe_offsets):
                stripes.append(
                    {
                        "angle_index": angle_index,
                        "view_yaw_deg": config.view_yaw_start_deg
                        + angle_index * config.view_yaw_step_deg,
                        "y0": stripe_y0,
                        "y1": stripe_y1,
                        "x0": x0,
                        "num_cols": num_cols,
                        "cell_size": cell_size,
                    }
                )
            blocks.append(
                {
                    "sequence_name": sequence_name,
                    "y0": region_y0,
                    "y1": region_y1,
                    "x0": x0,
                    "x1": x1,
                    "cell_size": cell_size,
                    "stripes": stripes,
                }
            )
    return blocks


def _blocks_from_config(blocks: list, config: SheetConfig) -> list[dict]:
    grouped: dict[str, dict] = {}
    for block in blocks:
        entry = grouped.setdefault(
            block.sequence,
            {
                "sequence_name": block.sequence,
                "y0": block.y0,
                "y1": block.y1,
                "x0": block.x0,
                "x1": block.x1,
                "cell_size": block.cell_size,
                "stripes": [],
            },
        )
        cell_size = block.cell_size or (block.y1 - block.y0 + 1)
        num_cols = (block.x1 - block.x0 + 1) // cell_size
        angle_index = len(entry["stripes"])
        entry["stripes"].append(
            {
                "angle_index": angle_index,
                "view_yaw_deg": config.view_yaw_start_deg + angle_index * config.view_yaw_step_deg,
                "y0": block.y0,
                "y1": block.y1,
                "x0": block.x0,
                "num_cols": num_cols,
                "cell_size": cell_size,
            }
        )
        entry["cell_size"] = cell_size
    result = list(grouped.values())
    for block in result:
        if len(block["stripes"]) != config.num_view_angles:
            raise ValueError(
                f"Sequence '{block['sequence_name']}' has {len(block['stripes'])} vertical angle "
                f"stripes; expected {config.num_view_angles}. Each sequence must have exactly "
                f"{config.num_view_angles} angle rows stacked vertically."
            )
    return result


def split_townsfolk_sheet(
    rgba: np.ndarray,
    output_dir: Path,
    config: SheetConfig,
) -> dict:
    from scripts.split_townsfolk import count_row_frames, extract_frame

    num_characters = config.num_characters or TOWNSFOLK_NUM_CHARACTERS
    frame_size = config.output_frame_size
    row_layout = detect_townsfolk_layout(rgba, num_characters, frame_size)

    manifest = {
        "source": str(config.input),
        "image_size": [rgba.shape[1], rgba.shape[0]],
        "frame_size": frame_size,
        "output_frame_size": frame_size,
        "grid_cell_size": frame_size,
        "grid_origin": [0, 0],
        **config.to_manifest_defaults(),
        "sheet_layout": "row_per_character",
        "characters": [],
        "blocks": [],
    }

    global_frame_index = 0
    for char_idx, (row_y, row_height) in enumerate(row_layout):
        frame_count = count_row_frames(rgba, row_y, row_height, frame_size)
        sequence_name = f"char_{char_idx:02d}"
        if config.characters and char_idx < len(config.characters):
            if config.characters[char_idx].sequences:
                sequence_name = config.characters[char_idx].sequences[0].name

        char_dir = output_dir / f"char_{char_idx:02d}"
        char_dir.mkdir(parents=True, exist_ok=True)
        char_entry = {
            "id": char_idx,
            "name": config.characters[char_idx].name if char_idx < len(config.characters) else sequence_name,
            "row_y": row_y,
            "row_height": row_height,
            "frame_count": frame_count,
            "sequence_name": sequence_name,
            "view_yaw_deg": 0.0,
            "angle_index": 0,
            "frames": [],
            "sequences": [
                {
                    "name": sequence_name,
                    "loop": (
                        config.characters[char_idx].sequences[0].loop
                        if char_idx < len(config.characters) and config.characters[char_idx].sequences
                        else None
                    ),
                    "standing": (
                        config.characters[char_idx].sequences[0].standing
                        if char_idx < len(config.characters) and config.characters[char_idx].sequences
                        else True
                    ),
                }
            ],
        }

        for frame_idx in range(frame_count):
            frame_x = frame_idx * frame_size
            frame_rgba = extract_frame(rgba, frame_x, row_y, frame_size, row_height)
            normalized = normalize_to_square(
                frame_rgba,
                frame_size,
                white_threshold=config.white_threshold,
            )
            if normalized is None:
                continue
            frame_name = f"frame_{global_frame_index:03d}.png"
            frame_path = char_dir / frame_name
            Image.fromarray(normalized.rgba).save(frame_path)

            char_entry["frames"].append(
                {
                    "index": global_frame_index,
                    "local_index": frame_idx,
                    "timestep_index": frame_idx,
                    "sequence_name": sequence_name,
                    "view_yaw_deg": 0.0,
                    "angle_index": 0,
                    "grid_cell_bbox": [frame_x, row_y, frame_x + frame_size, row_y + row_height],
                    "source_bbox": list(normalized.content_bbox),
                    "normalize_scale": normalized.normalize_scale,
                    "path": str(frame_path.relative_to(output_dir)),
                }
            )
            global_frame_index += 1

        manifest["characters"].append(char_entry)
        print(
            f"char_{char_idx:02d}: row_y={row_y} row_height={row_height} frames={frame_count}"
        )
    return manifest


def split_multi_angle_sheet(
    rgba: np.ndarray,
    output_dir: Path,
    config: SheetConfig,
) -> dict:
    blocks = detect_multi_angle_blocks(rgba, config)
    if not blocks:
        raise ValueError("No multi-angle blocks detected; provide explicit blocks[] in config")

    for angle_index in range(config.num_view_angles):
        for block in blocks:
            if len(block["stripes"]) > angle_index:
                block["stripes"][angle_index]["view_yaw_deg"] = (
                    config.view_yaw_start_deg + angle_index * config.view_yaw_step_deg
                )

    cell_size = blocks[0]["cell_size"] or blocks[0]["stripes"][0]["cell_size"]
    manifest = {
        "source": str(config.input),
        "image_size": [rgba.shape[1], rgba.shape[0]],
        "frame_size": config.output_frame_size,
        "output_frame_size": config.output_frame_size,
        "grid_cell_size": cell_size,
        "grid_origin": [0, 0],
        **config.to_manifest_defaults(),
        "sheet_layout": "multi_angle_blocks",
        "blocks": blocks,
        "characters": [],
    }

    character = config.characters[0] if config.characters else None
    char_id = character.id if character else 0
    char_name = character.name if character else "character_00"
    char_dir = output_dir / f"char_{char_id:02d}"
    char_dir.mkdir(parents=True, exist_ok=True)

    char_entry = {
        "id": char_id,
        "name": char_name,
        "frame_count": 0,
        "frames": [],
        "sequences": [
            {
                "name": sequence.name,
                "loop": sequence.loop,
                "standing": sequence.standing,
            }
            for sequence in (character.sequences if character else [])
        ],
    }

    global_frame_index = 0
    for block in blocks:
        sequence_name = block["sequence_name"]
        used_counts = []
        for stripe in block["stripes"]:
            used = count_used_cells(
                rgba,
                stripe["y0"],
                stripe["y1"],
                stripe["x0"],
                stripe["num_cols"],
                stripe["cell_size"],
                white_threshold=config.white_threshold,
            )
            stripe["used_cols"] = used
            used_counts.append(used)

        if len(set(used_counts)) != 1:
            print(
                f"WARNING: {sequence_name} angle rows disagree on used-cell count: {used_counts}"
            )
        num_timesteps = min(used_counts) if used_counts else 0

        for stripe in block["stripes"]:
            for timestep in range(num_timesteps):
                cell_x = stripe["x0"] + timestep * stripe["cell_size"]
                cell_y = stripe["y0"]
                cell_rgba = extract_cell(rgba, cell_x, cell_y, stripe["cell_size"])
                normalized = normalize_to_square(
                    cell_rgba,
                    config.output_frame_size,
                    white_threshold=config.white_threshold,
                )
                if normalized is None:
                    continue

                frame_name = f"frame_{global_frame_index:03d}.png"
                frame_path = char_dir / frame_name
                Image.fromarray(normalized.rgba).save(frame_path)

                char_entry["frames"].append(
                    {
                        "index": global_frame_index,
                        "local_index": len(char_entry["frames"]),
                        "timestep_index": timestep,
                        "sequence_name": sequence_name,
                        "view_yaw_deg": stripe["view_yaw_deg"],
                        "angle_index": stripe["angle_index"],
                        "grid_cell_bbox": [
                            cell_x,
                            cell_y,
                            cell_x + stripe["cell_size"],
                            cell_y + stripe["cell_size"],
                        ],
                        "source_bbox": list(normalized.content_bbox),
                        "normalize_scale": normalized.normalize_scale,
                        "path": str(frame_path.relative_to(output_dir)),
                    }
                )
                global_frame_index += 1

        print(
            f"{sequence_name}: block x={block['x0']}-{block['x1']} "
            f"timesteps={num_timesteps} angles={len(block['stripes'])}"
        )

    char_entry["frame_count"] = len(char_entry["frames"])
    manifest["characters"].append(char_entry)
    return manifest


def split_sprite_sheet_from_config(
    config: SheetConfig,
    dry_run: bool = False,
    preview_path: Path | None = None,
) -> dict:
    image = Image.open(config.input).convert("RGBA")
    rgba = np.array(image)
    output_dir = config.split_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    is_townsfolk = config.layout == "row_per_character" or (
        config.layout == "auto"
        and config.num_characters is not None
        and config.num_view_angles <= 1
    )

    if is_townsfolk:
        manifest = split_townsfolk_sheet(rgba, output_dir, config)
    else:
        blocks = detect_multi_angle_blocks(rgba, config)
        if preview_path is not None or dry_run:
            preview_target = preview_path or (output_dir / "grid_preview.png")
            draw_grid_preview(rgba, blocks, str(preview_target))
            print(f"Wrote grid preview: {preview_target}")
            for block in blocks:
                counts = [
                    count_used_cells(
                        rgba,
                        stripe["y0"],
                        stripe["y1"],
                        stripe["x0"],
                        stripe["num_cols"],
                        stripe["cell_size"],
                        white_threshold=config.white_threshold,
                    )
                    for stripe in block["stripes"]
                ]
                print(
                    f"  {block['sequence_name']}: angles={len(block['stripes'])} "
                    f"used_cells={counts}"
                )
        if dry_run:
            return {"blocks": blocks, "dry_run": True}
        manifest = split_multi_angle_sheet(rgba, output_dir, config)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print(f"Wrote manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a fixed-camera sprite sheet")
    parser.add_argument("--config", default=None, help="Sheet config JSON path")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--layout", default="auto")
    parser.add_argument("--num-characters", type=int, default=None)
    parser.add_argument("--output-frame-size", type=int, default=TOWNSFOLK_FRAME_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-output", default=None)
    args = parser.parse_args()

    if args.config:
        config = load_sheet_config(args.config)
    else:
        input_path = Path(args.input or "input/townsfolk.png")
        if not input_path.is_absolute():
            input_path = REPO_ROOT / input_path
        output_dir = Path(args.output or "output/split")
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        config = SheetConfig(
            input=input_path,
            output_root=output_dir.parent,
            split_dir=output_dir,
            layout=args.layout,
            output_frame_size=args.output_frame_size,
            num_characters=args.num_characters or TOWNSFOLK_NUM_CHARACTERS,
        )

    preview_path = None
    if args.preview_output:
        preview_path = Path(args.preview_output)
        if not preview_path.is_absolute():
            preview_path = REPO_ROOT / preview_path

    split_sprite_sheet_from_config(config, dry_run=args.dry_run, preview_path=preview_path)


if __name__ == "__main__":
    main()
