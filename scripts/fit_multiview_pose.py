#!/usr/bin/env python3
"""Joint per-timestep pose fitting across multiple view angles.

For sheets where frame index N is the same animation timestep viewed from
different camera azimuths, this stage fuses per-angle SAM3D estimates into one
shared body_pose/hand_pose per timestep while preserving per-angle global_rot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.manifest_utils import (
    group_frames_by_sequence_timestep,
    inference_frame_dir_index,
    load_manifest,
    load_selection_weights,
)
from scripts.mhr_repose import load_head_pose, repose
from scripts.pose_priors import weighted_median_per_dim

REFINE_STEPS = 5
REFINE_SCALE = 0.05


def load_pose(input_dir: Path, char_id: int, frame_info: dict) -> dict | None:
    frame_index = inference_frame_dir_index(frame_info)
    pose_path = input_dir / f"char_{char_id:02d}" / f"frame_{frame_index:03d}" / "pose.json"
    if not pose_path.exists():
        return None
    with open(pose_path, encoding="utf-8") as pose_file:
        return json.load(pose_file)


def fuse_timestep_poses(angle_poses: list[tuple[int, dict]], weights: list[float]) -> dict:
    weight_array = np.array(weights, dtype=np.float32)
    if weight_array.sum() <= 0:
        weight_array = np.ones(len(angle_poses), dtype=np.float32)
    body_matrix = np.array([pose["body_pose_params"] for _, pose in angle_poses], dtype=np.float32)
    hand_matrix = np.array([pose["hand_pose_params"] for _, pose in angle_poses], dtype=np.float32)
    fused_body = weighted_median_per_dim(body_matrix, weight_array)
    fused_hand = weighted_median_per_dim(hand_matrix, weight_array)
    best_idx = int(np.argmax(weight_array))
    best_angle, best_pose = angle_poses[best_idx]
    return {
        "body_pose_params": fused_body,
        "hand_pose_params": fused_hand,
        "shape_params": np.array(best_pose["shape_params"], dtype=np.float32),
        "scale_params": np.array(best_pose["scale_params"], dtype=np.float32),
        "expr_params": np.array(best_pose["expr_params"], dtype=np.float32),
        "seed_angle": best_angle,
    }


def refine_body_pose(
    head_pose,
    fused: dict,
    angle_poses: list[tuple[int, dict]],
    device: str,
) -> np.ndarray:
    body = fused["body_pose_params"].copy()
    best_score = -np.inf
    best_body = body.copy()
    for step in range(REFINE_STEPS):
        improved = False
        for dim in range(body.shape[0]):
            for delta in (-REFINE_SCALE, 0.0, REFINE_SCALE):
                trial = best_body.copy()
                trial[dim] += delta
                score = 0.0
                for angle_index, pose in angle_poses:
                    result = repose(
                        head_pose,
                        global_rot=np.array(pose["global_rot"], dtype=np.float32),
                        body_pose_params=trial,
                        hand_pose_params=fused["hand_pose_params"],
                        scale_params=fused["scale_params"],
                        shape_params=fused["shape_params"],
                        expr_params=fused["expr_params"],
                        device=device,
                    )
                    score += float(np.linalg.norm(result["pred_vertices"]))
                if score > best_score:
                    best_score = score
                    best_body = trial
                    improved = True
        if not improved:
            break
    return best_body


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-view per-timestep pose fusion")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--input-dir", default="output/inference_dp")
    parser.add_argument("--selection-report", default="output/inference_dp/selection_report.json")
    parser.add_argument("--output-dir", default="output/inference_multiview")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    selection_report_path = Path(args.selection_report)
    if not selection_report_path.is_absolute():
        selection_report_path = REPO_ROOT / selection_report_path

    manifest = load_manifest(split_dir)
    selection_weights = load_selection_weights(selection_report_path)

    head_pose = load_head_pose(device=args.device)
    summary = {"characters": []}

    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        timestep_groups = group_frames_by_sequence_timestep(char_entry)
        fused_count = 0
        skipped_sequences = []

        for (sequence_name, timestep), angle_frames in sorted(timestep_groups.items()):
            angle_indices = sorted(angle_frames.keys())
            expected_angles = manifest.get("num_view_angles", len(angle_indices))
            if len(angle_indices) != expected_angles:
                skipped_sequences.append(
                    {
                        "sequence": sequence_name,
                        "timestep": timestep,
                        "reason": f"angle_count_mismatch ({len(angle_indices)} != {expected_angles})",
                    }
                )
                continue
            if len(angle_indices) < 2:
                continue
            angle_poses = []
            weights = []
            for angle_index in angle_indices:
                frame_info = angle_frames[angle_index]
                pose = load_pose(input_dir, char_id, frame_info)
                if pose is None:
                    continue
                angle_poses.append((angle_index, pose))
                weights.append(
                    max(
                        selection_weights.get(char_id, {}).get(frame_info["index"], 1.0),
                        0.05,
                    )
                )
            if len(angle_poses) < 2:
                continue

            fused = fuse_timestep_poses(angle_poses, weights)
            fused_body = fused["body_pose_params"]

            for angle_index, pose in angle_poses:
                frame_info = angle_frames[angle_index]
                frame_index = inference_frame_dir_index(frame_info)
                result = repose(
                    head_pose,
                    global_rot=np.array(pose["global_rot"], dtype=np.float32),
                    body_pose_params=fused_body,
                    hand_pose_params=fused["hand_pose_params"],
                    scale_params=fused["scale_params"],
                    shape_params=fused["shape_params"],
                    expr_params=fused["expr_params"],
                    device=args.device,
                )
                new_pose = dict(pose)
                new_pose["body_pose_params"] = fused_body.tolist()
                new_pose["hand_pose_params"] = fused["hand_pose_params"].tolist()
                new_pose["pred_vertices"] = result["pred_vertices"].tolist()
                new_pose["pred_keypoints_3d"] = result["pred_keypoints_3d"].tolist()
                new_pose["pred_joint_coords"] = result["pred_joint_coords"].tolist()
                new_pose["pred_global_rots"] = result["pred_global_rots"].tolist()
                new_pose["mhr_model_params"] = result["mhr_model_params"].tolist()
                new_pose["multiview_fused"] = True
                new_pose["multiview_seed_angle"] = fused["seed_angle"]
                new_pose["sequence_name"] = sequence_name
                new_pose["timestep_index"] = timestep
                new_pose["angle_index"] = angle_index
                new_pose["view_yaw_deg"] = frame_info.get("view_yaw_deg", 0.0)

                frame_out_dir = output_dir / f"char_{char_id:02d}" / f"frame_{frame_index:03d}"
                frame_out_dir.mkdir(parents=True, exist_ok=True)
                with open(frame_out_dir / "pose.json", "w", encoding="utf-8") as pose_file:
                    json.dump(new_pose, pose_file, indent=2)
                fused_count += 1

        for frame_info in char_entry["frames"]:
            frame_index = inference_frame_dir_index(frame_info)
            if (output_dir / f"char_{char_id:02d}" / f"frame_{frame_index:03d}" / "pose.json").exists():
                continue
            pose = load_pose(input_dir, char_id, frame_info)
            if pose is None:
                continue
            frame_out_dir = output_dir / f"char_{char_id:02d}" / f"frame_{frame_index:03d}"
            frame_out_dir.mkdir(parents=True, exist_ok=True)
            with open(frame_out_dir / "pose.json", "w", encoding="utf-8") as pose_file:
                json.dump(pose, pose_file, indent=2)

        summary["characters"].append(
            {"char_id": char_id, "fused_frames": fused_count, "skipped_sequences": skipped_sequences}
        )
        print(f"char_{char_id:02d}: fused {fused_count} multi-view frames")

    with open(output_dir / "multiview_summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


if __name__ == "__main__":
    main()
