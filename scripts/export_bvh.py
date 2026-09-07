#!/usr/bin/env python3
"""Export per-character BVH animation files from SAM3D inference output."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam_3d_body.exporters.bvh_exporter import BVHExporter
from scripts.manifest_utils import inference_frame_dir_index

DEFAULT_FPS = 8.0
METERS_TO_CENTIMETERS = 100.0
DEFAULT_MHR_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"


def build_exporter(mhr_model=None, mhr_path: Path | str | None = None) -> BVHExporter:
    """SAM skeleton only: target_skeleton_path=None keeps the model's 127-joint hierarchy."""
    if mhr_model is not None:
        return BVHExporter(model_instance=mhr_model, target_skeleton_path=None)
    return BVHExporter(model_path=str(mhr_path or DEFAULT_MHR_PATH), target_skeleton_path=None)


def write_bvh(
    frame_poses: list[dict],
    output_path: Path,
    exporter: BVHExporter,
    fps: float = DEFAULT_FPS,
) -> None:
    if not frame_poses:
        return

    for pose in frame_poses:
        if "pred_joint_coords" not in pose:
            raise KeyError(
                "pose.json must include pred_joint_coords for BVH export; "
                f"frame_index={pose.get('frame_index', '?')}"
            )

    rots = np.stack([
        np.array(pose["pred_global_rots"], dtype=np.float32)
        for pose in frame_poses
    ])
    root = np.stack([
        np.array(pose["pred_joint_coords"], dtype=np.float32)[0]
        for pose in frame_poses
    ]) * METERS_TO_CENTIMETERS

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exporter.export(rots, root, str(output_path), frame_time=1.0 / fps)


def load_character_frames(inference_dir: Path, char_id: int) -> list[dict]:
    char_dir = inference_dir / f"char_{char_id:02d}"
    if not char_dir.exists():
        return []

    frames = []
    for frame_dir in sorted(char_dir.glob("frame_*")):
        pose_path = frame_dir / "pose.json"
        if pose_path.exists():
            with open(pose_path, encoding="utf-8") as pose_file:
                frames.append(json.load(pose_file))
    return frames


def load_sequence_frames(
    inference_dir: Path,
    char_id: int,
    frame_infos: list[dict],
) -> list[dict]:
    frames = []
    for frame_info in frame_infos:
        frame_idx = inference_frame_dir_index(frame_info)
        pose_path = inference_dir / f"char_{char_id:02d}" / f"frame_{frame_idx:03d}" / "pose.json"
        if not pose_path.exists():
            continue
        with open(pose_path, encoding="utf-8") as pose_file:
            pose_data = json.load(pose_file)
        pose_data["_frame_index"] = frame_idx
        pose_data["_timestep_index"] = int(frame_info.get("timestep_index", frame_idx))
        frames.append(pose_data)
    frames.sort(key=lambda item: item["_timestep_index"])
    return frames


def export_all_bvh(
    inference_dir: Path,
    output_dir: Path,
    exporter: BVHExporter,
    fps: float = DEFAULT_FPS,
    char_start: int = 0,
    char_end: int = 9,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for char_id in range(char_start, char_end):
        frames = load_character_frames(inference_dir, char_id)
        if not frames:
            print(f"char_{char_id:02d}: no frames, skipping")
            continue
        output_path = output_dir / f"char_{char_id:02d}.bvh"
        write_bvh(frames, output_path, exporter, fps=fps)
        print(f"char_{char_id:02d}: exported {len(frames)} frames -> {output_path}")


def export_sequence_bvh(
    inference_dir: Path,
    output_dir: Path,
    manifest: dict,
    exporter: BVHExporter,
    fps: float = DEFAULT_FPS,
    reference_angle: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        sequences: dict[str, list[dict]] = {}
        for frame_info in char_entry["frames"]:
            sequence_name = frame_info.get("sequence_name", "default")
            angle_index = int(frame_info.get("angle_index", 0))
            if angle_index != reference_angle:
                continue
            sequences.setdefault(sequence_name, []).append(frame_info)

        for sequence_name, frame_infos in sorted(sequences.items()):
            frame_infos.sort(key=lambda item: int(item.get("timestep_index", item["index"])))
            frames = load_sequence_frames(inference_dir, char_id, frame_infos)
            if not frames:
                print(f"char_{char_id:02d}/{sequence_name}: no frames, skipping")
                continue
            output_path = output_dir / f"char_{char_id:02d}_{sequence_name}.bvh"
            write_bvh(frames, output_path, exporter, fps=fps)
            print(f"char_{char_id:02d}/{sequence_name}: exported {len(frames)} frames -> {output_path}")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Export BVH animations from inference output")
    parser.add_argument("--inference-dir", default="output/inference")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--output-dir", default="output/bvh")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--char-start", type=int, default=0)
    parser.add_argument("--char-end", type=int, default=9)
    parser.add_argument(
        "--group-by-sequence",
        action="store_true",
        help="Export one BVH per (character, sequence) using reference angle 0",
    )
    parser.add_argument("--reference-angle", type=int, default=0)
    parser.add_argument("--mhr-path", default=str(DEFAULT_MHR_PATH))
    args = parser.parse_args()

    inference_dir = resolve_path(args.inference_dir)
    output_dir = resolve_path(args.output_dir)
    mhr_path = resolve_path(args.mhr_path)

    exporter = build_exporter(mhr_path=mhr_path)

    if args.group_by_sequence:
        split_dir = resolve_path(args.split_dir)
        with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        export_sequence_bvh(
            inference_dir,
            output_dir,
            manifest,
            exporter,
            fps=args.fps,
            reference_angle=args.reference_angle,
        )
        return

    export_all_bvh(
        inference_dir,
        output_dir,
        exporter,
        fps=args.fps,
        char_start=args.char_start,
        char_end=args.char_end,
    )


if __name__ == "__main__":
    main()
