#!/usr/bin/env python3
"""Helpers for reading manifest.json and grouping frames by pipeline stage."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_manifest(split_dir: Path) -> dict:
    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def inference_frame_dir_index(frame_info: dict) -> int:
    """Directory index under char_XX/frame_YYY used by inference output."""
    if "local_index" in frame_info:
        return int(frame_info["local_index"])
    return int(frame_info.get("timestep_index", frame_info["index"]))


def load_selection_weights(selection_report_path: Path) -> dict[int, dict[int, float]]:
    """Per-character, per-global-frame-index -> winning candidate score."""
    if not selection_report_path.exists():
        return {}
    with open(selection_report_path, encoding="utf-8") as report_file:
        report = json.load(report_file)
    weights: dict[int, dict[int, float]] = {}
    for char_report in report.get("characters", []):
        char_id = char_report["char_id"]
        weights[char_id] = {}
        for group in char_report.get("groups", []):
            for frame_report in group.get("frames", []):
                selected = frame_report["selected"]
                score = frame_report["scores"].get(selected)
                weights[char_id][frame_report["frame_index"]] = (
                    float(score) if score is not None else 0.0
                )
        for frame_report in char_report.get("frames", []):
            selected = frame_report["selected"]
            score = frame_report["scores"].get(selected)
            weights[char_id][frame_report["frame_index"]] = (
                float(score) if score is not None else 0.0
            )
    return weights


def frame_info_map(manifest: dict) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    for char_entry in manifest["characters"]:
        for frame_info in char_entry["frames"]:
            mapping[int(frame_info["index"])] = frame_info
    return mapping


def group_frames_by_sequence_angle(char_entry: dict) -> dict[tuple[str, int], list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for frame_info in char_entry["frames"]:
        key = (
            frame_info.get("sequence_name", "default"),
            int(frame_info.get("angle_index", 0)),
        )
        groups[key].append(frame_info)
    for key in groups:
        groups[key].sort(key=lambda frame: int(frame["timestep_index"]))
    return groups


def group_frames_by_sequence_timestep(char_entry: dict) -> dict[tuple[str, int], dict[int, dict]]:
    """Map (sequence_name, timestep_index) -> {angle_index: frame_info}."""
    groups: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for frame_info in char_entry["frames"]:
        sequence_name = frame_info.get("sequence_name", "default")
        timestep = int(frame_info.get("timestep_index", frame_info["index"]))
        angle_index = int(frame_info.get("angle_index", 0))
        groups[(sequence_name, timestep)][angle_index] = frame_info
    return groups


def sequence_meta(char_entry: dict, sequence_name: str) -> dict:
    for sequence in char_entry.get("sequences", []):
        if sequence["name"] == sequence_name:
            return sequence
    return {"name": sequence_name, "loop": False, "standing": True}


def non_standing_sequence_keys(manifest: dict, identity_report: dict | None = None) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    if identity_report is None:
        return keys
    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        char_report = identity_report.get("characters", {}).get(str(char_id), {})
        for sequence_name, sequence_report in char_report.get("sequences", {}).items():
            if not sequence_report.get("upright_constraint_applied", True):
                keys.add((char_id, sequence_name))
    return keys
