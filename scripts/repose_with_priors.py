#!/usr/bin/env python3
"""Stage 2 (upright constraint) + Stage 3 (locked identity), applied together
since both edit inputs to the same mhr_forward re-pose call.

Reads pose.json from --input-dir (the Stage 4 DP-selected poses), and for
every frame:
  1. Replaces global_rot with the upright-constrained version: the character's
     own up axis is snapped onto one shared "up" direction estimated (via a
     trimmed spherical mean, robust to per-frame estimator noise) across every
     frame of every character, while whatever yaw/twist that frame's original
     global_rot had is preserved (see scripts/pose_priors.py).
  2. Replaces shape_params/scale_params with one IoU-weighted-median value per
     character (computed once from all of that character's frames), so bone
     lengths are constant across the whole animation -- required for
     scripts/export_bvh.py to produce a valid rigid skeleton.
  3. Re-runs MHRHead.mhr_forward with the edited global_rot/shape_params/
     scale_params (body_pose_params/hand_pose_params/expr_params unchanged) to
     regenerate self-consistent pred_vertices/pred_keypoints_3d/
     pred_joint_coords/pred_global_rots/mhr_model_params.

Output geometry is intentionally left un-grounded (ground_offset reset to 0):
grounding is redone from scratch in Stage 6 (shared ground plane + bounded
per-frame nudge) using the fitted shared camera, so it isn't worth computing
twice here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.manifest_utils import inference_frame_dir_index, load_selection_weights  # noqa: E402
from scripts.mhr_repose import load_head_pose, repose  # noqa: E402
from scripts.pose_priors import (  # noqa: E402
    apply_upright_constraint,
    local_up_in_world,
    euler_zyx_to_rotmat,
    robust_shared_up,
    weighted_median_per_dim,
)

MIN_FRAME_WEIGHT = 0.05

# A character whose own frames' local-up-in-world deviates this much (degrees,
# averaged over the character) from the shared-up direction is not "standing
# with noisy per-frame tilt" (the case the constraint is designed for -- see
# char_06's up-to-38-degree per-frame std in the plan) but something
# genuinely non-upright for its whole animation (e.g. a lying-down death
# pose). Forcing the shared tilt onto such a character breaks its pose
# instead of denoising it. Empirically this cleanly separates every character
# in this sheet: the most-tilted still-standing/seated character is 31 degrees
# (char_02, seated) while the one genuinely-prone character is 75 degrees
# (char_04, death animation) -- see git history for the measurement.
NON_STANDING_MEAN_DEVIATION_DEG = 50.0


def load_manifest(split_dir: Path) -> dict:
    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def load_all_poses(input_dir: Path, manifest: dict) -> dict[int, list[dict]]:
    frame_meta = {}
    for char_entry in manifest["characters"]:
        for frame_info in char_entry["frames"]:
            frame_meta[(char_entry["id"], frame_info["index"])] = frame_info

    poses_by_char: dict[int, list[dict]] = {}
    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        frames = []
        for frame_info in char_entry["frames"]:
            pose_path = input_dir / f"char_{char_id:02d}" / f"frame_{inference_frame_dir_index(frame_info):03d}" / "pose.json"
            if not pose_path.exists():
                continue
            with open(pose_path, encoding="utf-8") as pose_file:
                pose_data = json.load(pose_file)
            pose_data["_frame_index"] = frame_info["index"]
            pose_data["_local_index"] = inference_frame_dir_index(frame_info)
            pose_data["_sequence_name"] = frame_info.get("sequence_name", f"char_{char_id:02d}")
            pose_data["_sequence_meta"] = sequence_meta_for_frame(char_entry, frame_info)
            frames.append(pose_data)
        poses_by_char[char_id] = frames
    return poses_by_char


def sequence_meta_for_frame(char_entry: dict, frame_info: dict) -> dict:
    sequence_name = frame_info.get("sequence_name")
    for sequence in char_entry.get("sequences", []):
        if sequence["name"] == sequence_name:
            return sequence
    return {"name": sequence_name or "default", "loop": False, "standing": True}


def estimate_global_shared_up(poses_by_char: dict[int, list[dict]]) -> np.ndarray:
    all_up_vectors = []
    for frames in poses_by_char.values():
        for pose_data in frames:
            rotmat = euler_zyx_to_rotmat(np.array(pose_data["global_rot"], dtype=np.float32))
            all_up_vectors.append(local_up_in_world(rotmat))
    return robust_shared_up(np.array(all_up_vectors, dtype=np.float32))


def mean_up_deviation_deg(frames: list[dict], shared_up: np.ndarray) -> float:
    """This character's own mean local-up-in-world deviation from shared_up (degrees)."""
    ups = []
    for pose_data in frames:
        rotmat = euler_zyx_to_rotmat(np.array(pose_data["global_rot"], dtype=np.float32))
        ups.append(local_up_in_world(rotmat))
    ups = np.array(ups, dtype=np.float32)
    cos_angles = np.clip(ups @ shared_up, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angles)).mean())


def compute_locked_identity(
    frames: list[dict], frame_weights: dict[int, float]
) -> tuple[np.ndarray, np.ndarray]:
    shape_matrix = np.array([f["shape_params"] for f in frames], dtype=np.float32)
    scale_matrix = np.array([f["scale_params"] for f in frames], dtype=np.float32)
    weights = np.array(
        [max(frame_weights.get(f["_frame_index"], 1.0), MIN_FRAME_WEIGHT) for f in frames],
        dtype=np.float32,
    )
    locked_shape = weighted_median_per_dim(shape_matrix, weights)
    locked_scale = weighted_median_per_dim(scale_matrix, weights)
    return locked_shape, locked_scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the shared-up orientation prior and locked per-character identity"
    )
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--input-dir", default="output/inference_dp")
    parser.add_argument("--selection-report", default="output/inference_dp/selection_report.json")
    parser.add_argument("--output-dir", default="output/inference_reposed")
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
    poses_by_char = load_all_poses(input_dir, manifest)
    selection_weights = load_selection_weights(selection_report_path)

    print("Estimating global shared-up direction across all frames...")
    shared_up = estimate_global_shared_up(poses_by_char)
    print(f"  shared up (world/camera space): {shared_up}")

    print("Loading MHR head for re-posing...")
    head_pose = load_head_pose(device=args.device)

    identity_report = {}
    sequence_standing: dict[tuple[int, str], bool] = {}
    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        frames = poses_by_char.get(char_id, [])
        if not frames:
            continue

        sequences_in_char: dict[str, list[dict]] = {}
        for pose_data in frames:
            sequence_name = pose_data["_sequence_name"]
            sequences_in_char.setdefault(sequence_name, []).append(pose_data)

        for sequence_name, sequence_frames in sequences_in_char.items():
            config_hint = sequence_frames[0]["_sequence_meta"].get("standing", True)
            deviation_deg = mean_up_deviation_deg(sequence_frames, shared_up)
            auto_non_standing = deviation_deg > NON_STANDING_MEAN_DEVIATION_DEG
            is_non_standing = auto_non_standing or not config_hint
            if config_hint and auto_non_standing:
                print(
                    f"  char_{char_id:02d}/{sequence_name}: config says standing but "
                    f"measured mean up-deviation {deviation_deg:.1f}deg -- treating as non-standing"
                )
            sequence_standing[(char_id, sequence_name)] = not is_non_standing

        frame_weights = selection_weights.get(char_id, {})
        locked_shape, locked_scale = compute_locked_identity(frames, frame_weights)
        identity_report[char_id] = {
            "shape_params": locked_shape.tolist(),
            "scale_params": locked_scale.tolist(),
            "num_frames": len(frames),
            "sequences": {
                sequence_name: {
                    "mean_up_deviation_deg": mean_up_deviation_deg(sequence_frames, shared_up),
                    "upright_constraint_applied": sequence_standing[(char_id, sequence_name)],
                }
                for sequence_name, sequence_frames in sequences_in_char.items()
            },
        }

        char_output_dir = output_dir / f"char_{char_id:02d}"
        for pose_data in frames:
            frame_idx = pose_data["_local_index"]
            sequence_name = pose_data["_sequence_name"]
            apply_upright = sequence_standing[(char_id, sequence_name)]
            original_global_rot = np.array(pose_data["global_rot"], dtype=np.float32)
            upright_global_rot = (
                original_global_rot
                if not apply_upright
                else apply_upright_constraint(original_global_rot, shared_up)
            )

            result = repose(
                head_pose,
                global_rot=upright_global_rot,
                body_pose_params=np.array(pose_data["body_pose_params"], dtype=np.float32),
                hand_pose_params=np.array(pose_data["hand_pose_params"], dtype=np.float32),
                scale_params=locked_scale,
                shape_params=locked_shape,
                expr_params=np.array(pose_data["expr_params"], dtype=np.float32),
                device=args.device,
            )

            new_pose_data = dict(pose_data)
            del new_pose_data["_frame_index"]
            del new_pose_data["_sequence_name"]
            del new_pose_data["_sequence_meta"]
            new_pose_data["global_rot"] = upright_global_rot.tolist()
            new_pose_data["shape_params"] = locked_shape.tolist()
            new_pose_data["scale_params"] = locked_scale.tolist()
            new_pose_data["pred_vertices"] = result["pred_vertices"].tolist()
            new_pose_data["pred_keypoints_3d"] = result["pred_keypoints_3d"].tolist()
            new_pose_data["pred_joint_coords"] = result["pred_joint_coords"].tolist()
            new_pose_data["pred_global_rots"] = result["pred_global_rots"].tolist()
            new_pose_data["mhr_model_params"] = result["mhr_model_params"].tolist()
            new_pose_data["ground_offset"] = 0.0

            frame_out_dir = char_output_dir / f"frame_{frame_idx:03d}"
            frame_out_dir.mkdir(parents=True, exist_ok=True)
            with open(frame_out_dir / "pose.json", "w", encoding="utf-8") as pose_file:
                json.dump(new_pose_data, pose_file, indent=2)

        print(f"char_{char_id:02d}: reposed {len(frames)} frames with locked identity")

    identity_report_path = output_dir / "locked_identity_report.json"
    identity_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(identity_report_path, "w", encoding="utf-8") as report_file:
        json.dump({"shared_up": shared_up.tolist(), "characters": identity_report}, report_file, indent=2)
    print(f"Wrote: {identity_report_path}")


if __name__ == "__main__":
    main()
