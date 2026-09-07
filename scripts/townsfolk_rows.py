#!/usr/bin/env python3
"""Emit per-character crop rects for the townsfolk sprite sheet.

Each character occupies one row band, and sprites sit on a fixed cell pitch
(FRAME_SIZE) starting at x=0, with only the first N cells used. The crop width
must therefore be N * FRAME_SIZE -- cropping the full sheet width and slicing
it into N columns only lines up for rows that happen to use every cell.

Output is one pipe-delimited line per character, for shell consumption:

    name|x|y|width|height|frames|loop
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

from scripts.split_townsfolk import (  # noqa: E402
    FRAME_SIZE,
    count_row_frames,
    detect_row_layout,
    is_white_mask,
)
from scripts.sprite_grid import DEFAULT_WHITE_THRESHOLD  # noqa: E402

DEFAULT_INPUT = "input/townsfolk.png"
DEFAULT_CONFIG = "configs/townsfolk.json"
# A row band's outer edges can include the sheet's white separator lines (at
# the top for most rows, at the bottom for char_00). Sprite pixels are never
# white, so any edge row above this white fraction is a separator to trim.
WHITE_ROW_FRACTION_THRESHOLD = 0.05


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def trim_white_edge_rows(band_rgba: np.ndarray) -> tuple[int, int]:
    """Return (top_offset, height) with white separator rows trimmed off both edges."""
    height = band_rgba.shape[0]
    white_fractions = [
        float(is_white_mask(band_rgba[y, :, :3].astype(np.int32), DEFAULT_WHITE_THRESHOLD).mean())
        for y in range(height)
    ]

    top = 0
    while top < height and white_fractions[top] >= WHITE_ROW_FRACTION_THRESHOLD:
        top += 1

    bottom = height - 1
    while bottom > top and white_fractions[bottom] >= WHITE_ROW_FRACTION_THRESHOLD:
        bottom -= 1

    return top, bottom - top + 1


def character_rows(input_path: Path, config_path: Path) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    characters = config["characters"]
    sheet = np.array(Image.open(input_path).convert("RGBA"))
    layout = detect_row_layout(sheet, len(characters))

    rows = []
    for character, (row_top, row_height) in zip(characters, layout):
        frame_count = count_row_frames(sheet, row_top, row_height, FRAME_SIZE)
        if frame_count <= 0:
            continue

        width = frame_count * FRAME_SIZE
        band = sheet[row_top : row_top + row_height, :width]
        top_offset, height = trim_white_edge_rows(band)

        rows.append(
            {
                "name": character["name"],
                "x": 0,
                "y": row_top + top_offset,
                "width": width,
                "height": height,
                "frames": frame_count,
                "loop": bool(character["sequences"][0].get("loop", True)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    rows = character_rows(resolve_path(args.input), resolve_path(args.config))
    for row in rows:
        print(
            f"{row['name']}|{row['x']}|{row['y']}|{row['width']}|"
            f"{row['height']}|{row['frames']}|{int(row['loop'])}"
        )


if __name__ == "__main__":
    main()
