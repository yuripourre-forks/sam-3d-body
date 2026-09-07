#!/usr/bin/env python3
"""Export per-character BVH animation files from SAM3D inference output."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.manifest_utils import inference_frame_dir_index

# (name, parent_index, start_keypoint, end_keypoint) using MHR70 indices
BVH_JOINTS = [
    ("Hips", -1, 9, 10),
    ("Spine", 0, 5, 6),
    ("Neck", 1, 69, 69),
    ("Head", 2, 0, 0),
    ("LeftUpLeg", 0, 9, 11),
    ("LeftLeg", 4, 11, 13),
    ("LeftFoot", 5, 13, 17),
    ("RightUpLeg", 0, 10, 12),
    ("RightLeg", 7, 12, 14),
    ("RightFoot", 8, 14, 20),
    ("LeftArm", 1, 5, 7),
    ("LeftForeArm", 10, 7, 9),
    ("LeftHand", 11, 9, 41),
    ("RightArm", 1, 6, 8),
    ("RightForeArm", 13, 8, 10),
    ("RightHand", 14, 10, 42),
]

DEFAULT_FPS = 8.0


def rotation_matrix_to_euler_zyx(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy >= 1e-6:
        x = math.atan2(rotation[2, 1], rotation[2, 2])
        y = math.atan2(-rotation[2, 0], sy)
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x = math.atan2(-rotation[1, 2], rotation[1, 1])
        y = math.atan2(-rotation[2, 0], sy)
        z = 0.0
    return math.degrees(z), math.degrees(y), math.degrees(x)


def build_skeleton_positions(keypoints_3d: np.ndarray) -> list[np.ndarray]:
    positions = []
    for _, parent_idx, start_idx, end_idx in BVH_JOINTS:
        if parent_idx == -1:
            pos = (keypoints_3d[start_idx] + keypoints_3d[end_idx]) / 2.0
        elif start_idx == end_idx:
            pos = keypoints_3d[start_idx]
        else:
            pos = keypoints_3d[end_idx]
        positions.append(pos)
    return positions


def compute_offsets(rest_positions: list[np.ndarray]) -> list[np.ndarray]:
    offsets = []
    for joint_idx, (_, parent_idx, _, _) in enumerate(BVH_JOINTS):
        if parent_idx < 0:
            offsets.append(rest_positions[joint_idx].copy())
        else:
            offsets.append(rest_positions[joint_idx] - rest_positions[parent_idx])
    return offsets


def get_joint_rotation(
    global_rots: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    rot_idx = end_idx if start_idx != end_idx else start_idx
    rot_idx = min(max(rot_idx, 0), global_rots.shape[0] - 1)
    return global_rots[rot_idx]


def write_joint(
    lines: list[str],
    joint_idx: int,
    offsets: list[np.ndarray],
    depth: int,
) -> None:
    """Write one JOINT/ROOT block, including its opening/closing braces.

    A prior version of this function never wrote the opening `{` (only the
    closing `}`), producing a HIERARCHY with unmatched braces. Compliant BVH
    parsers (e.g. Python's `bvh` package) desync on that malformed structure
    and can end up trying to parse a header token like "Xposition" as a
    motion-data float. Standard BVH requires, for every ROOT/JOINT *and* every
    End Site, a `{`/`}`-delimited block:

        JOINT Name
        {
          OFFSET x y z
          CHANNELS n ...
          <children or End Site>
        }
    """
    name, parent_idx, _, _ = BVH_JOINTS[joint_idx]
    indent = "  " * depth
    child_indent = "  " * (depth + 1)
    offset = offsets[joint_idx]

    if parent_idx < 0:
        lines.append("ROOT Hips")
    else:
        lines.append(f"{indent}JOINT {name}")
    lines.append(f"{indent}{{")

    lines.append(
        f"{child_indent}OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}"
    )
    if parent_idx < 0:
        lines.append(
            f"{child_indent}CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
        )
    else:
        lines.append(f"{child_indent}CHANNELS 3 Zrotation Yrotation Xrotation")

    children = [
        child_idx
        for child_idx, (_, child_parent, _, _) in enumerate(BVH_JOINTS)
        if child_parent == joint_idx
    ]
    if not children:
        end_site_indent = "  " * (depth + 2)
        lines.append(f"{child_indent}End Site")
        lines.append(f"{child_indent}{{")
        lines.append(f"{end_site_indent}OFFSET 0.000000 0.050000 0.000000")
        lines.append(f"{child_indent}}}")
    else:
        for child_idx in children:
            write_joint(lines, child_idx, offsets, depth + 1)
    lines.append(f"{indent}}}")


def write_bvh(
    frame_poses: list[dict],
    output_path: Path,
    fps: float = DEFAULT_FPS,
) -> None:
    if not frame_poses:
        return

    rest_keypoints = np.array(frame_poses[0]["pred_keypoints_3d"], dtype=np.float32)
    rest_positions = build_skeleton_positions(rest_keypoints)
    offsets = compute_offsets(rest_positions)

    hierarchy_lines = ["HIERARCHY"]
    write_joint(hierarchy_lines, 0, offsets, 0)

    motion_lines = ["MOTION", f"Frames: {len(frame_poses)}", f"Frame Time: {1.0 / fps:.6f}"]
    for frame_pose in frame_poses:
        keypoints_3d = np.array(frame_pose["pred_keypoints_3d"], dtype=np.float32)
        global_rots = np.array(frame_pose["pred_global_rots"], dtype=np.float32)
        positions = build_skeleton_positions(keypoints_3d)

        frame_values = []
        for joint_idx, (_, parent_idx, start_idx, end_idx) in enumerate(BVH_JOINTS):
            rot_mat = get_joint_rotation(global_rots, start_idx, end_idx)
            z_rot, y_rot, x_rot = rotation_matrix_to_euler_zyx(rot_mat)
            if parent_idx < 0:
                frame_values.extend(positions[joint_idx].tolist())
            frame_values.extend([z_rot, y_rot, x_rot])
        motion_lines.append(" ".join(f"{value:.6f}" for value in frame_values))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as bvh_file:
        bvh_file.write("\n".join(hierarchy_lines) + "\n")
        bvh_file.write("\n".join(motion_lines) + "\n")


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
        write_bvh(frames, output_path, fps=fps)
        print(f"char_{char_id:02d}: exported {len(frames)} frames -> {output_path}")


def export_sequence_bvh(
    inference_dir: Path,
    output_dir: Path,
    manifest: dict,
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
            write_bvh(frames, output_path, fps=fps)
            print(f"char_{char_id:02d}/{sequence_name}: exported {len(frames)} frames -> {output_path}")


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
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    if not inference_dir.is_absolute():
        inference_dir = REPO_ROOT / inference_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    if args.group_by_sequence:
        split_dir = Path(args.split_dir)
        if not split_dir.is_absolute():
            split_dir = REPO_ROOT / split_dir
        with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        export_sequence_bvh(
            inference_dir,
            output_dir,
            manifest,
            fps=args.fps,
            reference_angle=args.reference_angle,
        )
        return

    export_all_bvh(
        inference_dir,
        output_dir,
        fps=args.fps,
        char_start=args.char_start,
        char_end=args.char_end,
    )


if __name__ == "__main__":
    main()
