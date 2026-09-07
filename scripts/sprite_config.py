#!/usr/bin/env python3
"""Load and validate per-sheet pipeline configuration JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WHITE_THRESHOLD = 240
DEFAULT_WHITE_FRACTION_THRESHOLD = 0.98
DEFAULT_OUTPUT_FRAME_SIZE = 96
DEFAULT_NUM_VIEW_ANGLES = 8
DEFAULT_VIEW_YAW_STEP_DEG = 45.0
DEFAULT_VIEW_YAW_START_DEG = 0.0
DEFAULT_CANDIDATE_UPSCALERS = ["hypir4x"]
DEFAULT_LARGE_SHEET_FRAME_THRESHOLD = 200
DEFAULT_BOUNDARY_LAMBDA = 0.002


@dataclass
class SequenceConfig:
    name: str
    loop: bool = False
    standing: bool = True


@dataclass
class CharacterConfig:
    id: int
    name: str
    sequences: list[SequenceConfig] = field(default_factory=list)


@dataclass
class BlockOverride:
    sequence: str
    y0: int
    y1: int
    x0: int
    x1: int
    cell_size: int | None = None


@dataclass
class SheetConfig:
    input: Path
    output_root: Path
    split_dir: Path
    layout: str = "auto"
    output_frame_size: int = DEFAULT_OUTPUT_FRAME_SIZE
    white_threshold: int = DEFAULT_WHITE_THRESHOLD
    white_fraction_threshold: float = DEFAULT_WHITE_FRACTION_THRESHOLD
    num_view_angles: int = DEFAULT_NUM_VIEW_ANGLES
    view_yaw_step_deg: float = DEFAULT_VIEW_YAW_STEP_DEG
    view_yaw_start_deg: float = DEFAULT_VIEW_YAW_START_DEG
    candidate_upscalers: list[str] = field(default_factory=lambda: list(DEFAULT_CANDIDATE_UPSCALERS))
    large_sheet_frame_threshold: int = DEFAULT_LARGE_SHEET_FRAME_THRESHOLD
    boundary_lambda: float = DEFAULT_BOUNDARY_LAMBDA
    num_characters: int | None = None
    grid_cell_size: int | None = None
    characters: list[CharacterConfig] = field(default_factory=list)
    blocks: list[BlockOverride] = field(default_factory=list)
    device: str = "cuda"
    upscale: int = 4
    render_scale: int = 4

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    def sequence_meta(self, sequence_name: str) -> SequenceConfig | None:
        for character in self.characters:
            for sequence in character.sequences:
                if sequence.name == sequence_name:
                    return sequence
        return None

    def inference_dir(self, stage: str) -> Path:
        return self.output_root / stage

    def to_manifest_defaults(self) -> dict[str, Any]:
        return {
            "sheet_layout": self.layout,
            "ignore_white_lines": True,
            "white_threshold": self.white_threshold,
            "white_fraction_threshold": self.white_fraction_threshold,
            "output_frame_size": self.output_frame_size,
            "num_view_angles": self.num_view_angles,
            "view_yaw_step_deg": self.view_yaw_step_deg,
            "view_yaw_start_deg": self.view_yaw_start_deg,
            "boundary_lambda": self.boundary_lambda,
        }


def _parse_sequence(entry: dict) -> SequenceConfig:
    return SequenceConfig(
        name=str(entry["name"]),
        loop=bool(entry.get("loop", False)),
        standing=bool(entry.get("standing", True)),
    )


def _parse_character(entry: dict) -> CharacterConfig:
    return CharacterConfig(
        id=int(entry["id"]),
        name=str(entry.get("name", f"char_{entry['id']:02d}")),
        sequences=[_parse_sequence(seq) for seq in entry.get("sequences", [])],
    )


def _parse_block(entry: dict) -> BlockOverride:
    return BlockOverride(
        sequence=str(entry["sequence"]),
        y0=int(entry["y0"]),
        y1=int(entry["y1"]),
        x0=int(entry["x0"]),
        x1=int(entry["x1"]),
        cell_size=int(entry["cell_size"]) if entry.get("cell_size") is not None else None,
    )


def load_sheet_config(config_path: str | Path) -> SheetConfig:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as config_file:
        raw = json.load(config_file)

    output_root = Path(raw.get("output_root", "output/sheet"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    split_dir = raw.get("split_dir", str(output_root / "split"))
    split_path = Path(split_dir)
    if not split_path.is_absolute():
        split_path = REPO_ROOT / split_path

    input_path = Path(raw["input"])
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    return SheetConfig(
        input=input_path,
        output_root=output_root,
        split_dir=split_path,
        layout=str(raw.get("layout", "auto")),
        output_frame_size=int(raw.get("output_frame_size", DEFAULT_OUTPUT_FRAME_SIZE)),
        white_threshold=int(raw.get("white_threshold", DEFAULT_WHITE_THRESHOLD)),
        white_fraction_threshold=float(
            raw.get("white_fraction_threshold", DEFAULT_WHITE_FRACTION_THRESHOLD)
        ),
        num_view_angles=int(raw.get("num_view_angles", DEFAULT_NUM_VIEW_ANGLES)),
        view_yaw_step_deg=float(raw.get("view_yaw_step_deg", DEFAULT_VIEW_YAW_STEP_DEG)),
        view_yaw_start_deg=float(raw.get("view_yaw_start_deg", DEFAULT_VIEW_YAW_START_DEG)),
        candidate_upscalers=list(raw.get("candidate_upscalers", DEFAULT_CANDIDATE_UPSCALERS)),
        large_sheet_frame_threshold=int(
            raw.get("large_sheet_frame_threshold", DEFAULT_LARGE_SHEET_FRAME_THRESHOLD)
        ),
        boundary_lambda=float(raw.get("boundary_lambda", DEFAULT_BOUNDARY_LAMBDA)),
        num_characters=int(raw["num_characters"]) if raw.get("num_characters") is not None else None,
        grid_cell_size=int(raw["grid_cell_size"]) if raw.get("grid_cell_size") is not None else None,
        characters=[_parse_character(entry) for entry in raw.get("characters", [])],
        blocks=[_parse_block(entry) for entry in raw.get("blocks", [])],
        device=str(raw.get("device", "cuda")),
        upscale=int(raw.get("upscale", 4)),
        render_scale=int(raw.get("render_scale", 4)),
    )
