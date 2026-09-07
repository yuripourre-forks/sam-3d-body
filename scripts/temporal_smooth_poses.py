#!/usr/bin/env python3
"""Stage 5: outlier repair + temporal regularization, then re-pose.

Reads pose.json from --input-dir (the Stage 2/3 upright + locked-identity
output) and, per character:

  1. Flags frames whose pose distance (mean per-joint geodesic rotation angle
     between pred_global_rots) to *both* temporal neighbors exceeds a robust
     per-character threshold -- the "spike then return" signature of estimator
     noise rather than genuine motion -- and repairs them by interpolating
     between their two neighbors (SLERP on the global rotation, LERP on
     body_pose_params/hand_pose_params).
  2. Detects looping animations by comparing the first and last sprite frames'
     own silhouettes; when they're similar enough, treats the sequence as
     cyclic for both outlier neighbor lookup and smoothing.
  3. Applies mild Savitzky-Golay smoothing (window 5, order 2) to
     body_pose_params and to the sequence's yaw angle (the one remaining
     rotational degree of freedom after Stage 2's upright constraint pins
     every frame's tilt to the shared value) -- exactly the two places actual
     per-frame motion lives, so smoothing here can't accidentally erase real
     motion the way smoothing raw vertices/keypoints would.
  4. Re-poses every frame via MHRHead.mhr_forward with the repaired/smoothed
     parameters (identity stays locked from Stage 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.mhr_repose import load_head_pose, repose  # noqa: E402
from scripts.manifest_utils import group_frames_by_sequence_angle, inference_frame_dir_index, sequence_meta  # noqa: E402
from scripts.pose_metrics import content_mask, mask_iou  # noqa: E402
from scripts.pose_priors import (  # noqa: E402
    apply_yaw_angle,
    euler_zyx_to_rotmat,
    extract_yaw_angle,
    local_up_in_world,
    rotmat_to_euler_zyx,
)

import roma
import torch

OUTLIER_MAD_MULTIPLIER = 2.5
LOOP_IOU_THRESHOLD = 0.75
SAVGOL_WINDOW = 5
SAVGOL_ORDER = 2


def load_manifest(split_dir: Path) -> dict:
    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def load_character_frames(input_dir: Path, char_entry: dict) -> list[dict]:
    frames = []
    for frame_info in char_entry["frames"]:
        pose_path = input_dir / f"char_{char_entry['id']:02d}" / f"frame_{inference_frame_dir_index(frame_info):03d}" / "pose.json"
        if not pose_path.exists():
            continue
        with open(pose_path, encoding="utf-8") as pose_file:
            pose_data = json.load(pose_file)
        pose_data["_frame_index"] = frame_info["index"]
        pose_data["_local_index"] = inference_frame_dir_index(frame_info)
        pose_data["_sequence_name"] = frame_info.get("sequence_name", f"char_{char_entry['id']:02d}")
        pose_data["_angle_index"] = int(frame_info.get("angle_index", 0))
        pose_data["_timestep_index"] = int(frame_info.get("timestep_index", frame_info["index"]))
        frames.append(pose_data)
    return frames


def sequence_is_loop(char_entry: dict, sequence_name: str, split_dir: Path, frame_infos: list[dict]) -> bool:
    meta = sequence_meta(char_entry, sequence_name)
    if meta.get("loop") is not None:
        return bool(meta["loop"])
    if len(frame_infos) < 3:
        return False
    first_path = split_dir / frame_infos[0]["path"]
    last_path = split_dir / frame_infos[-1]["path"]
    first_mask = content_mask(np.array(Image.open(first_path).convert("RGBA")))
    last_mask = content_mask(np.array(Image.open(last_path).convert("RGBA")))
    iou = mask_iou(first_mask, last_mask)
    return iou is not None and iou >= LOOP_IOU_THRESHOLD


def detect_loop(split_dir: Path, char_entry: dict) -> bool:
    frame_infos = char_entry["frames"]
    if len(frame_infos) < 3:
        return False
    first_path = split_dir / frame_infos[0]["path"]
    last_path = split_dir / frame_infos[-1]["path"]
    first_mask = content_mask(np.array(Image.open(first_path).convert("RGBA")))
    last_mask = content_mask(np.array(Image.open(last_path).convert("RGBA")))
    iou = mask_iou(first_mask, last_mask)
    return iou is not None and iou >= LOOP_IOU_THRESHOLD


def root_rotation_geodesic_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic angle (degrees) between two single (3, 3) root rotation matrices."""
    relative = rot_a.T @ rot_b
    cos_angle = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def body_pose_l2_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """L2 distance between two frames' body_pose_params.

    Needed alongside root-rotation distance (see find_outliers()'s docstring):
    a raw single-frame estimate can have wildly wrong *limb* articulation
    (e.g. arms crossed overhead, legs together) while its root orientation
    happens to land close to a neighbor's -- or, for "standing" sequences,
    after this function's caller already ran on Stage 2/3's upright-corrected
    input, where every frame's root rotation has been forced to match the
    shared "up" direction regardless of how wrong the raw per-frame estimate
    was, which erases the rotation-distance signal for exactly this failure
    mode while leaving the bad limb pose untouched.
    """
    return float(np.linalg.norm(pose_a - pose_b))


def consecutive_distances(
    global_rots: list[np.ndarray], body_poses: list[np.ndarray], is_loop: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Root-rotation geodesic angle and body_pose_params L2 distance between
    each pair of temporal neighbors.

    Returns two arrays of length T (loop) or T-1 (non-loop): distances[i] is
    the distance between frame i and frame i+1 (wrapping to frame 0 for the
    last entry when is_loop is True).
    """
    num_frames = len(global_rots)
    pairs = range(num_frames) if is_loop else range(num_frames - 1)
    rotation_distances = []
    body_pose_distances = []
    for i in pairs:
        j = (i + 1) % num_frames
        rotation_distances.append(root_rotation_geodesic_angle_deg(global_rots[i], global_rots[j]))
        body_pose_distances.append(body_pose_l2_distance(body_poses[i], body_poses[j]))
    return np.array(rotation_distances, dtype=np.float32), np.array(body_pose_distances, dtype=np.float32)


def _mad_spike_frames(distances: np.ndarray, num_frames: int, is_loop: bool) -> set[int]:
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median))) or 1e-6
    threshold = median + OUTLIER_MAD_MULTIPLIER * 1.4826 * mad

    def dist_to_next(i):
        return distances[i] if i < len(distances) else None

    def dist_from_prev(i):
        prev = (i - 1) % num_frames if is_loop else i - 1
        if not is_loop and prev < 0:
            return None
        return distances[prev]

    spikes = set()
    frame_range = range(num_frames) if is_loop else range(1, num_frames - 1)
    for i in frame_range:
        prev_dist = dist_from_prev(i)
        next_dist = dist_to_next(i)
        if prev_dist is not None and next_dist is not None and prev_dist > threshold and next_dist > threshold:
            spikes.add(i)
    return spikes


def find_outliers(
    rotation_distances: np.ndarray, body_pose_distances: np.ndarray, num_frames: int, is_loop: bool
) -> set[int]:
    """Flag frames whose distance to *both* temporal neighbors spikes above a
    robust per-sequence threshold in *either* the root-rotation or body-pose
    metric -- the "spike then return" signature of single-frame estimator
    noise rather than genuine motion. Both metrics are needed: see
    body_pose_l2_distance()'s docstring for why rotation alone isn't enough.
    """
    return _mad_spike_frames(rotation_distances, num_frames, is_loop) | _mad_spike_frames(
        body_pose_distances, num_frames, is_loop
    )


def slerp_rotmat(rot_a: np.ndarray, rot_b: np.ndarray, t: float) -> np.ndarray:
    tensor_a = torch.as_tensor(rot_a, dtype=torch.float32)
    tensor_b = torch.as_tensor(rot_b, dtype=torch.float32)
    steps = torch.tensor([t], dtype=torch.float32)
    return roma.rotmat_slerp(tensor_a, tensor_b, steps)[0].numpy()


def find_outlier_runs(outlier_indices: set[int], num_frames: int, is_loop: bool) -> list[list[int]]:
    """Group outlier indices into maximal contiguous runs (wrapping across the
    loop boundary for loop sequences). A single bad frame inflates the
    neighbor-distance of its own neighbors too (see find_outliers()), so
    real-world outlier sets are often short runs of 2-3 consecutive indices,
    not always isolated single frames.
    """
    if not outlier_indices:
        return []
    sorted_indices = sorted(outlier_indices)
    runs = [[sorted_indices[0]]]
    for idx in sorted_indices[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    if (
        is_loop
        and len(runs) > 1
        and runs[0][0] == 0
        and runs[-1][-1] == num_frames - 1
    ):
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    return runs


def repair_outliers(
    frames: list[dict], outlier_indices: set[int], is_loop: bool
) -> None:
    """In-place repair: interpolate global_rot (SLERP) and body/hand pose (LERP)
    across each run of consecutive outlier frames, using the nearest good
    frames just outside the run on each side -- not necessarily each outlier
    frame's immediate neighbor, since that neighbor may itself be part of the
    same outlier run (find_outlier_runs()). Frames within the run are placed
    at their proportional position between those two good boundary frames
    (t=step/num_steps), which reduces to the original fixed t=0.5 for
    single-frame runs.
    """
    num_frames = len(frames)
    runs = find_outlier_runs(outlier_indices, num_frames, is_loop)
    for run in runs:
        run_start, run_end = run[0], run[-1]
        prev_i = (run_start - 1) % num_frames
        next_i = (run_end + 1) % num_frames
        touches_non_loop_boundary = not is_loop and (run_start == 0 or run_end == num_frames - 1)
        if touches_non_loop_boundary or prev_i in outlier_indices or next_i in outlier_indices:
            # No clean frame exists on one (or both) sides of the run -- e.g. the run
            # touches a non-loop sequence's boundary, or (degenerately) every frame in
            # the sequence is flagged -- so skip repair rather than compounding error
            # by interpolating from another still-bad frame.
            continue

        prev_rotmat = euler_zyx_to_rotmat(np.array(frames[prev_i]["global_rot"], dtype=np.float32))
        next_rotmat = euler_zyx_to_rotmat(np.array(frames[next_i]["global_rot"], dtype=np.float32))
        prev_body_hand = {
            key: np.array(frames[prev_i][key], dtype=np.float32) for key in ("body_pose_params", "hand_pose_params")
        }
        next_body_hand = {
            key: np.array(frames[next_i][key], dtype=np.float32) for key in ("body_pose_params", "hand_pose_params")
        }
        num_steps = (next_i - prev_i) % num_frames

        for i in run:
            step = (i - prev_i) % num_frames
            t = step / num_steps
            interpolated_rotmat = slerp_rotmat(prev_rotmat, next_rotmat, t)
            frames[i]["global_rot"] = rotmat_to_euler_zyx(interpolated_rotmat).tolist()
            for key in ("body_pose_params", "hand_pose_params"):
                frames[i][key] = ((1.0 - t) * prev_body_hand[key] + t * next_body_hand[key]).tolist()


def savgol_smooth_series(values: np.ndarray, is_loop: bool) -> np.ndarray:
    window = min(SAVGOL_WINDOW, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < SAVGOL_ORDER + 2:
        return values
    mode = "wrap" if is_loop else "interp"
    if values.ndim == 1:
        return savgol_filter(values, window_length=window, polyorder=SAVGOL_ORDER, mode=mode)
    return savgol_filter(values, window_length=window, polyorder=SAVGOL_ORDER, mode=mode, axis=0)


def smooth_character(frames: list[dict], is_loop: bool) -> None:
    """In-place: Savitzky-Golay smoothing of body_pose_params and the yaw-about-
    shared-up angle series (see scripts/pose_priors.py's extract/apply_yaw_angle).
    """
    if len(frames) < SAVGOL_ORDER + 2:
        return

    rotmats = [euler_zyx_to_rotmat(np.array(f["global_rot"], dtype=np.float32)) for f in frames]
    # Every frame's local-up-in-world is already pinned to the same shared_up by
    # Stage 2, so it can be recovered directly from any single frame's rotation.
    shared_up = local_up_in_world(rotmats[0])
    reference_rotmat = rotmats[0]
    yaw_angles = np.array(
        [extract_yaw_angle(rotmat, shared_up, reference_rotmat) for rotmat in rotmats], dtype=np.float32
    )
    yaw_angles_unwrapped = np.unwrap(yaw_angles)
    smoothed_yaw = savgol_smooth_series(yaw_angles_unwrapped, is_loop)

    body_pose_matrix = np.array([f["body_pose_params"] for f in frames], dtype=np.float32)
    smoothed_body_pose = savgol_smooth_series(body_pose_matrix, is_loop)

    for i, frame in enumerate(frames):
        new_rotmat = apply_yaw_angle(float(smoothed_yaw[i]), shared_up, reference_rotmat)
        frame["global_rot"] = rotmat_to_euler_zyx(new_rotmat).tolist()
        frame["body_pose_params"] = smoothed_body_pose[i].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Outlier repair + temporal smoothing, then re-pose")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--input-dir", default="output/inference_reposed")
    parser.add_argument("--output-dir", default="output/inference_smoothed")
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

    manifest = load_manifest(split_dir)
    head_pose = load_head_pose(device=args.device)

    summary = {}
    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        all_frames = load_character_frames(input_dir, char_entry)
        frame_lookup = {frame["_frame_index"]: frame for frame in all_frames}
        groups = group_frames_by_sequence_angle(char_entry)
        outlier_indices: set[int] = set()

        for (sequence_name, angle_index), frame_infos in groups.items():
            frames = []
            for frame_info in frame_infos:
                pose = frame_lookup.get(frame_info["index"])
                if pose is not None:
                    frames.append(pose)
            frames.sort(key=lambda item: item["_timestep_index"])
            if len(frames) < 3:
                continue
            is_loop = sequence_is_loop(char_entry, sequence_name, split_dir, frame_infos)
            global_rots = [euler_zyx_to_rotmat(np.array(f["global_rot"], dtype=np.float32)) for f in frames]
            body_poses = [np.array(f["body_pose_params"], dtype=np.float32) for f in frames]
            rotation_distances, body_pose_distances = consecutive_distances(global_rots, body_poses, is_loop)
            group_outliers = find_outliers(rotation_distances, body_pose_distances, len(frames), is_loop)
            if group_outliers:
                print(
                    f"char_{char_id:02d}/{sequence_name}/a{angle_index}: repairing "
                    f"{len(group_outliers)} outlier frame(s) {sorted(group_outliers)}"
                )
                repair_outliers(frames, group_outliers, is_loop)
                outlier_indices.update(frames[index]["_frame_index"] for index in group_outliers)
            smooth_character(frames, is_loop)
            print(
                f"char_{char_id:02d}/{sequence_name}/a{angle_index}: loop={is_loop}, "
                f"smoothed {len(frames)} frames"
            )

        char_output_dir = output_dir / f"char_{char_id:02d}"
        for frame in all_frames:
            frame_idx = frame["_local_index"]
            result = repose(
                head_pose,
                global_rot=np.array(frame["global_rot"], dtype=np.float32),
                body_pose_params=np.array(frame["body_pose_params"], dtype=np.float32),
                hand_pose_params=np.array(frame["hand_pose_params"], dtype=np.float32),
                scale_params=np.array(frame["scale_params"], dtype=np.float32),
                shape_params=np.array(frame["shape_params"], dtype=np.float32),
                expr_params=np.array(frame["expr_params"], dtype=np.float32),
                device=args.device,
            )

            new_frame = dict(frame)
            del new_frame["_frame_index"]
            new_frame["pred_vertices"] = result["pred_vertices"].tolist()
            new_frame["pred_keypoints_3d"] = result["pred_keypoints_3d"].tolist()
            new_frame["pred_joint_coords"] = result["pred_joint_coords"].tolist()
            new_frame["pred_global_rots"] = result["pred_global_rots"].tolist()
            new_frame["mhr_model_params"] = result["mhr_model_params"].tolist()

            frame_out_dir = char_output_dir / f"frame_{frame_idx:03d}"
            frame_out_dir.mkdir(parents=True, exist_ok=True)
            with open(frame_out_dir / "pose.json", "w", encoding="utf-8") as pose_file:
                json.dump(new_frame, pose_file, indent=2)

        summary[char_id] = {"num_frames": len(all_frames), "repaired_outliers": sorted(outlier_indices)}

    summary_path = output_dir / "temporal_smoothing_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
