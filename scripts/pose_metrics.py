#!/usr/bin/env python3
"""Pose-fidelity metrics that separate "wrong pose" from "different clothing".

Raw silhouette IoU (as used by compare_mesh_render_reassembly.py /
select_best_pose.py) has a ceiling: the mesh is nude while sprites wear robes
and dresses, so a character in a full-length robe/dress saturates at a modest
IoU no matter how correct the pose is (the legs never match). This script adds
three metrics per scripts/../.cursor/plans/fixed_camera_sequence_priors_*.plan.md
Stage 0 that are much less sensitive to clothing and much more sensitive to
actual pose/orientation errors:

- core-region IoU: silhouette IoU restricted to the top CORE_HEIGHT_FRACTION of
  the character's own sprite bounding box (torso + head), which is where a
  robe/dress differs least from a nude mesh and where "tipped over" /
  handstand-style pose failures show up most clearly.
- foot-contact error: pixel distance between the mesh render's lowest opaque
  row and the sprite's own lowest content row, in the same (already-aligned)
  canvas coordinate space.
- temporal jerk: mean per-joint geodesic rotation angle between consecutive
  frames' pred_global_rots, reported next to the sprite's own mean consecutive
  silhouette IoU (a smoothness reference -- animations with high sprite-to-sprite
  IoU should produce low pose-to-pose jerk if the estimated poses are correct).

Used both to baseline the current output (before the fixed-camera/sequence-prior
work) and to verify improvement at the end (Stage 7).
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

from scripts.manifest_utils import group_frames_by_sequence_angle, inference_frame_dir_index  # noqa: E402
from scripts.sprite_grid import content_mask as sprite_content_mask  # noqa: E402

DEFAULT_BOUNDARY_LAMBDA = 0.002

# Rough torso+head fraction of standing height (hips sit a little past the
# midpoint of total height for most human proportions); used only to cut the
# comparison canvas into a "core" band that excludes legs/robe/skirt, not as a
# precise anatomical claim.
CORE_HEIGHT_FRACTION = 0.55
FOOT_KEYPOINT_INDICES = [15, 16, 17, 18, 19, 20]


def content_mask(image_rgba: np.ndarray) -> np.ndarray:
    return sprite_content_mask(image_rgba)


def sprite_mask_at_size(frame_path: Path, size: tuple[int, int]) -> np.ndarray:
    """Original sprite frame's content mask, nearest-upscaled to `size` (w, h)."""
    source_rgba = Image.open(frame_path).convert("RGBA")
    upscaled = source_rgba.resize(size, Image.NEAREST)
    return content_mask(np.array(upscaled))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return None
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def mask_boundary_points(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    interior = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    boundary = interior & (neighbors < 4)
    ys, xs = np.where(boundary)
    if ys.size == 0:
        ys, xs = np.where(mask)
    return np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)


def symmetric_chamfer_px(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    points_a = mask_boundary_points(mask_a)
    points_b = mask_boundary_points(mask_b)
    if points_a.size == 0 or points_b.size == 0:
        return None
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    forward = float(distances.min(axis=1).mean())
    backward = float(distances.min(axis=0).mean())
    return (forward + backward) / 2.0


def composite_score(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    boundary_lambda: float = DEFAULT_BOUNDARY_LAMBDA,
) -> float | None:
    iou = mask_iou(mask_a, mask_b)
    if iou is None:
        return None
    chamfer = symmetric_chamfer_px(mask_a, mask_b)
    if chamfer is None:
        return iou
    return float(iou - boundary_lambda * chamfer)


def mirror_consistency(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    """Compare mask_b to a horizontal flip of mask_a (diagnostic for angle pairs)."""
    flipped = np.fliplr(mask_a)
    return mask_iou(flipped, mask_b)


def mask_bottom_row(mask: np.ndarray) -> int | None:
    rows = np.where(mask.any(axis=1))[0]
    return int(rows.max()) if rows.size else None


def mask_top_row(mask: np.ndarray) -> int | None:
    rows = np.where(mask.any(axis=1))[0]
    return int(rows.min()) if rows.size else None


def core_row_cutoff(sprite_mask: np.ndarray, core_fraction: float = CORE_HEIGHT_FRACTION) -> int | None:
    """Row index below which content is considered "legs" and excluded from core IoU."""
    top = mask_top_row(sprite_mask)
    bottom = mask_bottom_row(sprite_mask)
    if top is None or bottom is None:
        return None
    return top + int(round((bottom - top + 1) * core_fraction))


def frame_metrics(sprite_mask: np.ndarray, mesh_mask: np.ndarray, boundary_lambda: float = DEFAULT_BOUNDARY_LAMBDA) -> dict:
    raw_iou = mask_iou(sprite_mask, mesh_mask)
    chamfer_px = symmetric_chamfer_px(sprite_mask, mesh_mask)
    composite = composite_score(sprite_mask, mesh_mask, boundary_lambda=boundary_lambda)

    cutoff = core_row_cutoff(sprite_mask)
    core_iou = None
    if cutoff is not None:
        core_iou = mask_iou(sprite_mask[:cutoff], mesh_mask[:cutoff])

    sprite_bottom = mask_bottom_row(sprite_mask)
    mesh_bottom = mask_bottom_row(mesh_mask)
    foot_contact_error_px = (
        float(abs(sprite_bottom - mesh_bottom)) if sprite_bottom is not None and mesh_bottom is not None else None
    )

    return {
        "raw_iou": raw_iou,
        "core_iou": core_iou,
        "chamfer_px": chamfer_px,
        "composite_score": composite,
        "foot_contact_error_px": foot_contact_error_px,
    }


def rotation_geodesic_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    """Per-joint geodesic angle (degrees) between two (J, 3, 3) rotation batches."""
    relative = np.einsum("jik,jkl->jil", np.transpose(rot_a, (0, 2, 1)), rot_b)
    trace = np.trace(relative, axis1=1, axis2=2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def character_temporal_jerk(pose_jsons: list[dict]) -> float | None:
    """Mean per-joint geodesic rotation angle between consecutive frames."""
    if len(pose_jsons) < 2:
        return None
    angles = []
    for prev_pose, curr_pose in zip(pose_jsons, pose_jsons[1:]):
        rot_prev = np.array(prev_pose["pred_global_rots"], dtype=np.float32)
        rot_curr = np.array(curr_pose["pred_global_rots"], dtype=np.float32)
        angles.append(rotation_geodesic_angle_deg(rot_prev, rot_curr).mean())
    return float(np.mean(angles))


def character_sprite_smoothness(split_dir: Path, char_entry: dict) -> float | None:
    """Mean consecutive-frame silhouette IoU of the character's own sprite frames.

    Used purely as a reference: an animation whose own artwork changes smoothly
    frame-to-frame (high IoU here) should not require large pose-to-pose jumps
    to reproduce, so a much lower temporal_jerk-vs-this-ratio indicates residual
    estimator noise rather than genuine motion.
    """
    frame_infos = char_entry["frames"]
    if len(frame_infos) < 2:
        return None
    masks = []
    for frame_info in frame_infos:
        frame_path = split_dir / frame_info["path"]
        rgba = np.array(Image.open(frame_path).convert("RGBA"))
        masks.append(content_mask(rgba))
    ious = [mask_iou(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
    ious = [iou for iou in ious if iou is not None]
    return float(np.mean(ious)) if ious else None


def mirror_consistency_for_character(char_entry: dict, split_dir: Path, inference_dir: Path) -> list[dict]:
    """Diagnostic: compare angle i vs angle (i+4) mod 8 mirror consistency."""
    num_angles = max(
        int(frame_info.get("angle_index", 0)) for frame_info in char_entry["frames"]
    ) + 1
    if num_angles < 2:
        return []

    groups = group_frames_by_sequence_angle(char_entry)
    pair_reports = []
    half_turn = num_angles // 2

    for (sequence_name, angle_index), frame_infos in groups.items():
        mirror_angle = (angle_index + half_turn) % num_angles
        mirror_frames = groups.get((sequence_name, mirror_angle))
        if not mirror_frames:
            continue

        mirror_lookup = {
            int(frame["timestep_index"]): frame for frame in mirror_frames
        }
        pair_ious = []
        for frame_info in frame_infos:
            timestep = int(frame_info.get("timestep_index", frame_info["index"]))
            mirror_info = mirror_lookup.get(timestep)
            if mirror_info is None:
                continue
            sprite_mask = content_mask(np.array(Image.open(split_dir / frame_info["path"]).convert("RGBA")))
            mirror_mask = content_mask(np.array(Image.open(split_dir / mirror_info["path"]).convert("RGBA")))
            mirror_score = mirror_consistency(sprite_mask, mirror_mask)
            if mirror_score is not None:
                pair_ious.append(mirror_score)

        if pair_ious:
            pair_reports.append(
                {
                    "sequence_name": sequence_name,
                    "angle_index": angle_index,
                    "mirror_angle_index": mirror_angle,
                    "mean_mirror_iou": float(np.mean(pair_ious)),
                }
            )
    return pair_reports


def evaluate_inference_dir(
    split_dir: Path,
    inference_dir: Path,
    manifest: dict,
) -> dict:
    per_character = []
    all_raw_ious: list[float] = []
    all_core_ious: list[float] = []
    all_foot_errors: list[float] = []

    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        frame_reports = []
        pose_jsons = []

        for frame_info in char_entry["frames"]:
            frame_idx = inference_frame_dir_index(frame_info)
            frame_dir = inference_dir / f"char_{char_id:02d}" / f"frame_{frame_idx:03d}"
            mesh_render_path = frame_dir / "mesh_render.png"
            pose_path = frame_dir / "pose.json"
            if not mesh_render_path.exists():
                continue

            mesh_rgba = np.array(Image.open(mesh_render_path).convert("RGBA"))
            mesh_mask = mesh_rgba[..., 3] > 0
            sprite_mask = sprite_mask_at_size(
                split_dir / frame_info["path"], (mesh_rgba.shape[1], mesh_rgba.shape[0])
            )

            metrics = frame_metrics(sprite_mask, mesh_mask)
            metrics["frame_index"] = frame_idx
            frame_reports.append(metrics)

            if metrics["raw_iou"] is not None:
                all_raw_ious.append(metrics["raw_iou"])
            if metrics["core_iou"] is not None:
                all_core_ious.append(metrics["core_iou"])
            if metrics["foot_contact_error_px"] is not None:
                all_foot_errors.append(metrics["foot_contact_error_px"])

            if pose_path.exists():
                with open(pose_path, encoding="utf-8") as pose_file:
                    pose_jsons.append(json.load(pose_file))

        def _mean(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None

        per_character.append(
            {
                "char_id": char_id,
                "mean_raw_iou": _mean([f["raw_iou"] for f in frame_reports]),
                "mean_core_iou": _mean([f["core_iou"] for f in frame_reports]),
                "mean_composite_score": _mean([f["composite_score"] for f in frame_reports]),
                "mean_chamfer_px": _mean([f["chamfer_px"] for f in frame_reports]),
                "mean_foot_contact_error_px": _mean([f["foot_contact_error_px"] for f in frame_reports]),
                "temporal_jerk_deg": character_temporal_jerk(pose_jsons),
                "sprite_smoothness_iou": character_sprite_smoothness(split_dir, char_entry),
                "mirror_consistency": mirror_consistency_for_character(
                    char_entry, split_dir, inference_dir
                ),
                "frames": frame_reports,
            }
        )

    return {
        "inference_dir": str(inference_dir),
        "overall_mean_raw_iou": float(np.mean(all_raw_ious)) if all_raw_ious else None,
        "overall_mean_core_iou": float(np.mean(all_core_ious)) if all_core_ious else None,
        "overall_mean_foot_contact_error_px": float(np.mean(all_foot_errors)) if all_foot_errors else None,
        "overall_mean_temporal_jerk_deg": float(
            np.mean([c["temporal_jerk_deg"] for c in per_character if c["temporal_jerk_deg"] is not None])
        )
        if any(c["temporal_jerk_deg"] is not None for c in per_character)
        else None,
        "characters": per_character,
    }


def print_summary(label: str, report: dict) -> None:
    print(f"\n=== {label}: {report['inference_dir']} ===")
    print(f"  overall mean raw IoU:          {report['overall_mean_raw_iou']:.4f}")
    print(f"  overall mean core IoU:         {report['overall_mean_core_iou']:.4f}")
    print(f"  overall mean foot-contact err: {report['overall_mean_foot_contact_error_px']:.2f} px")
    print(f"  overall mean temporal jerk:    {report['overall_mean_temporal_jerk_deg']:.2f} deg")
    for char_report in report["characters"]:
        jerk = char_report["temporal_jerk_deg"]
        smoothness = char_report["sprite_smoothness_iou"]
        jerk_str = f"{jerk:.2f}deg" if jerk is not None else "n/a"
        smoothness_str = f"{smoothness:.3f}" if smoothness is not None else "n/a"
        print(
            f"    char_{char_report['char_id']:02d}: "
            f"raw_iou={char_report['mean_raw_iou']:.3f} "
            f"core_iou={char_report['mean_core_iou']:.3f} "
            f"foot_err={char_report['mean_foot_contact_error_px']:.1f}px "
            f"jerk={jerk_str} (sprite smoothness iou={smoothness_str})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Core-IoU / foot-contact / temporal-jerk pose metrics")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument(
        "--inference-dir",
        action="append",
        dest="inference_dirs",
        default=None,
        help="Inference directory to evaluate; repeatable to compare multiple (e.g. before/after)",
    )
    parser.add_argument("--report", default="output/pose_metrics_report.json")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir

    inference_dirs = args.inference_dirs or ["output/inference"]

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path

    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    reports = {}
    for inference_dir_arg in inference_dirs:
        inference_dir = Path(inference_dir_arg)
        if not inference_dir.is_absolute():
            inference_dir = REPO_ROOT / inference_dir
        report = evaluate_inference_dir(split_dir, inference_dir, manifest)
        reports[inference_dir_arg] = report
        print_summary(inference_dir_arg, report)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(reports, report_file, indent=2)
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
