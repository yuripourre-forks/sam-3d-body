#!/usr/bin/env python3
"""Split townsfolk sprite sheet into per-character frame sequences.

Thin wrapper around scripts/split_sprite_sheet.py for backward compatibility.
Core grid utilities live in scripts/sprite_grid.py; townsfolk-specific row
layout detection remains here because variable row heights are unique to that
sheet format.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.sprite_grid import (
    MAGENTA_RGBA,
    MIN_MAGENTA_RATIO,
    MIN_CONTENT_BAND_SIZE as MIN_ROW_BAND_HEIGHT,
    count_used_cells,
    is_magenta_mask,
    is_white_mask,
    normalize_to_square,
)

NUM_CHARACTERS = 9
FRAME_SIZE = 96
WHITE_THRESHOLD = 240

__all__ = [
    "NUM_CHARACTERS",
    "FRAME_SIZE",
    "MAGENTA_RGBA",
    "WHITE_THRESHOLD",
    "MIN_MAGENTA_RATIO",
    "MIN_ROW_BAND_HEIGHT",
    "is_magenta_mask",
    "is_white_mask",
    "detect_character_bands",
    "detect_row_layout",
    "extract_frame",
    "count_row_frames",
    "split_sprite_sheet",
]


def detect_character_bands(
    sheet_rgba: np.ndarray,
    num_rows: int,
) -> list[tuple[int, int]]:
    sheet_height = sheet_rgba.shape[0]
    character_mask = content_mask_from_rgba(sheet_rgba)
    row_has_content = character_mask.any(axis=1)

    bands = []
    in_band = False
    start = 0
    for y in range(sheet_height):
        if row_has_content[y] and not in_band:
            start = y
            in_band = True
        elif not row_has_content[y] and in_band:
            bands.append((start, y - 1))
            in_band = False
    if in_band:
        bands.append((start, sheet_height - 1))

    bands = [band for band in bands if band[1] - band[0] + 1 >= MIN_ROW_BAND_HEIGHT]
    if len(bands) < num_rows:
        raise ValueError(
            f"Detected only {len(bands)} character row bands, expected {num_rows}"
        )
    return bands[:num_rows]


def content_mask_from_rgba(sheet_rgba: np.ndarray) -> np.ndarray:
    rgb = sheet_rgba[..., :3].astype(np.int32)
    return (
        ~is_magenta_mask(rgb)
        & ~is_white_mask(rgb, WHITE_THRESHOLD)
        & (sheet_rgba[..., 3] > 0)
    )


def detect_row_layout(
    sheet_rgba: np.ndarray,
    num_rows: int,
    default_row_height: int = FRAME_SIZE,
) -> list[tuple[int, int]]:
    sheet_height = sheet_rgba.shape[0]
    bands = detect_character_bands(sheet_rgba, num_rows)

    deficit = num_rows * default_row_height - sheet_height
    if deficit <= 0:
        return [(index * default_row_height, default_row_height) for index in range(num_rows)]

    short_height = default_row_height - deficit

    for short_idx in range(num_rows):
        heights = [default_row_height] * num_rows
        heights[short_idx] = short_height

        tops = []
        cursor = 0
        for height in heights:
            tops.append(cursor)
            cursor += height

        if cursor != sheet_height:
            continue

        layout_valid = True
        for row_index, (band_start, band_end) in enumerate(bands):
            row_top = tops[row_index]
            row_bottom = row_top + heights[row_index]
            if band_start < row_top or band_end >= row_bottom:
                layout_valid = False
                break

        if layout_valid:
            return [(tops[index], heights[index]) for index in range(num_rows)]

    raise ValueError(
        f"Could not find a valid row layout for {num_rows} rows "
        f"in a {sheet_height}px-tall sheet"
    )


def extract_frame(
    sheet_rgba: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    sheet_height, sheet_width = sheet_rgba.shape[:2]
    frame = np.tile(MAGENTA_RGBA, (height, width, 1))

    src_x1 = min(x + width, sheet_width)
    src_y1 = min(y + height, sheet_height)
    if x < sheet_width and y < sheet_height:
        crop = sheet_rgba[y:src_y1, x:src_x1]
        frame[: crop.shape[0], : crop.shape[1]] = crop

    return frame


def count_row_frames(
    sheet_rgba: np.ndarray,
    row_top: int,
    row_height: int,
    frame_width: int = FRAME_SIZE,
) -> int:
    return count_used_cells(
        sheet_rgba,
        row_top,
        row_top + row_height - 1,
        0,
        (sheet_rgba.shape[1] + frame_width - 1) // frame_width,
        frame_width,
        white_threshold=WHITE_THRESHOLD,
        min_magenta_ratio=MIN_MAGENTA_RATIO,
    )


def split_sprite_sheet(
    input_path: str,
    output_dir: str,
    num_characters: int = NUM_CHARACTERS,
    frame_size: int = FRAME_SIZE,
) -> dict:
    from scripts.sprite_config import SheetConfig
    from scripts.split_sprite_sheet import split_sprite_sheet_from_config

    config = SheetConfig(
        input=Path(input_path),
        output_root=Path(output_dir).parent,
        split_dir=Path(output_dir),
        layout="row_per_character",
        output_frame_size=frame_size,
        num_characters=num_characters,
    )
    return split_sprite_sheet_from_config(config)


def main():
    parser = argparse.ArgumentParser(description="Split townsfolk sprite sheet")
    parser.add_argument("--input", default="input/townsfolk.png")
    parser.add_argument("--output", default="output/split")
    parser.add_argument("--num-characters", type=int, default=NUM_CHARACTERS)
    parser.add_argument("--frame-size", type=int, default=FRAME_SIZE)
    parser.add_argument("--config", default="configs/townsfolk.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.config:
        from scripts.sprite_config import load_sheet_config
        from scripts.split_sprite_sheet import split_sprite_sheet_from_config

        config = load_sheet_config(repo_root / args.config)
        if args.input != "input/townsfolk.png":
            config.input = repo_root / args.input if not Path(args.input).is_absolute() else Path(args.input)
        if args.output != "output/split":
            config.split_dir = repo_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
        split_sprite_sheet_from_config(config)
        return

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    split_sprite_sheet(str(input_path), str(output_dir), args.num_characters, args.frame_size)


if __name__ == "__main__":
    main()
