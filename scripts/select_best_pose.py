#!/usr/bin/env python3
"""Pick the best SAM3D-Body pose per frame from several candidate inference runs.

SAM3D-Body's pose estimate for these small, heavily-padded pixel-art crops turns
out to be quite sensitive to *how* the frame was upscaled before inference: some
poses are recovered correctly with a diffusion-based upscale (HYPIR) at 4x, some
only at 8x, and some only with a plain deterministic Lanczos resize -- no single
upscale strategy is a strict improvement over the others across all 9 characters.

Selection is a sequence-aware Viterbi/DP pass over each character's row of
frames (Stage 4 of the fixed-camera-and-sequence-priors plan), rather than an
independent per-frame argmax: each candidate's per-frame silhouette IoU against
the sprite's own mask is an emission reward, and the mean per-joint geodesic
rotation angle between consecutive frames' pred_global_rots is a transition
penalty, so the path picked prefers candidates that are *both* individually
well-fit *and* consistent with their neighbors -- avoiding, e.g., picking a
locally-high-IoU candidate that requires a 60-degree pose jump from the frame
before and after it (a hallmark of estimator noise, not real motion). No
sequence segmentation is needed: the sprites' own consecutive-frame silhouette
IoU never drops below ~0.84 with no detected hard cuts, so each character row
is safely treated as one smooth sequence end to end.

Each candidate's saved focal_length is also normalized to a canonical upscale
factor before being copied out, fixing the bimodal-intrinsics defect noted in
the investigation this script came out of (hypir4x/lanczos4x run SAM3D-Body at
4x, hypir8x at 8x, so raw focal_length values cluster at two different scales
depending purely on which candidate a given frame happened to select).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.manifest_utils import group_frames_by_sequence_angle, inference_frame_dir_index  # noqa: E402
from scripts.pose_metrics import (  # noqa: E402
    composite_score,
    content_mask,
    mask_iou,
    rotation_geodesic_angle_deg,
)

CANDIDATE_FILES = ("pose.json", "mesh.ply", "mesh_render.png", "mesh_iso.png")
BOUNDARY_LAMBDA = 0.002

# Upscale factor SAM3D-Body was actually run at for each candidate strategy (see
# process_townsfolk.py's --upscale); focal_length is renormalized to CANONICAL_UPSCALE
# so it stops depending on which strategy a frame's winning candidate happened to be.
CANDIDATE_UPSCALE = {"hypir4x": 4, "hypir8x": 8, "lanczos4x": 4}
CANONICAL_UPSCALE = 4

# lambda (rotation-jerk penalty weight, IoU-per-degree) search grid for tuning the
# DP's smoothness/fit trade-off per character; see select_lambda_for_character().
LAMBDA_GRID = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
IOU_DEGRADATION_TOLERANCE = 0.03


def sprite_silhouette(frame_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """Original sprite frame's content mask, nearest-upscaled to the render size."""
    source_rgba = Image.open(frame_path).convert("RGBA")
    upscaled = source_rgba.resize(target_size, Image.NEAREST)
    return content_mask(np.array(upscaled))


def load_candidate_frame_data(
    candidate_root: Path, char_id: int, frame_idx: int, frame_path: Path
) -> dict | None:
    frame_dir = candidate_root / f"char_{char_id:02d}" / f"frame_{frame_idx:03d}"
    mesh_render_path = frame_dir / "mesh_render.png"
    pose_path = frame_dir / "pose.json"
    if not mesh_render_path.exists() or not pose_path.exists():
        return None

    mesh_rgba = np.array(Image.open(mesh_render_path).convert("RGBA"))
    mesh_mask = mesh_rgba[..., 3] > 0
    source_mask = sprite_silhouette(frame_path, (mesh_rgba.shape[1], mesh_rgba.shape[0]))
    score = composite_score(mesh_mask, source_mask, boundary_lambda=BOUNDARY_LAMBDA) or 0.0
    iou = mask_iou(mesh_mask, source_mask) or 0.0

    with open(pose_path, encoding="utf-8") as pose_file:
        pose_data = json.load(pose_file)

    return {
        "frame_dir": frame_dir,
        "score": score,
        "iou": iou,
        "global_rots": np.array(pose_data["pred_global_rots"], dtype=np.float32),
    }


def dp_select_path(
    score_matrix: np.ndarray,
    rotation_candidates: list[list[np.ndarray | None]],
    lambda_weight: float,
) -> list[int]:
    """Viterbi DP: maximize sum_t score(c_t) - lambda * sum_t angle(pose(c_t), pose(c_t-1)).

    score_matrix: (T, C) with -inf for missing candidates.
    rotation_candidates[t][c]: (127, 3, 3) global joint rotations, or None if missing.
    Returns the selected candidate index per frame (T,).
    """
    num_frames, num_candidates = score_matrix.shape
    value = np.full((num_frames, num_candidates), -np.inf)
    backptr = np.zeros((num_frames, num_candidates), dtype=int)
    value[0] = score_matrix[0]

    for t in range(1, num_frames):
        for c in range(num_candidates):
            if not np.isfinite(score_matrix[t, c]):
                continue
            best_prev_score, best_prev_c = -np.inf, 0
            for prev_c in range(num_candidates):
                if not np.isfinite(value[t - 1, prev_c]):
                    continue
                penalty = 0.0
                if rotation_candidates[t][c] is not None and rotation_candidates[t - 1][prev_c] is not None:
                    penalty = lambda_weight * float(
                        rotation_geodesic_angle_deg(
                            rotation_candidates[t - 1][prev_c], rotation_candidates[t][c]
                        ).mean()
                    )
                score = value[t - 1, prev_c] - penalty
                if score > best_prev_score:
                    best_prev_score, best_prev_c = score, prev_c
            value[t, c] = best_prev_score + score_matrix[t, c]
            backptr[t, c] = best_prev_c

    last_c = int(np.argmax(value[-1]))
    path = [last_c]
    for t in range(num_frames - 1, 0, -1):
        last_c = backptr[t, path[-1]]
        path.append(int(last_c))
    path.reverse()
    return path


def path_metrics(
    path: list[int], score_matrix: np.ndarray, rotation_candidates: list[list[np.ndarray | None]]
) -> tuple[float, float | None]:
    scores = [score_matrix[t, c] for t, c in enumerate(path) if np.isfinite(score_matrix[t, c])]
    mean_score = float(np.mean(scores)) if scores else 0.0

    angles = []
    for t in range(1, len(path)):
        prev_rot = rotation_candidates[t - 1][path[t - 1]]
        curr_rot = rotation_candidates[t][path[t]]
        if prev_rot is not None and curr_rot is not None:
            angles.append(rotation_geodesic_angle_deg(prev_rot, curr_rot).mean())
    jerk = float(np.mean(angles)) if angles else None
    return mean_score, jerk


def select_lambda_for_character(
    score_matrix: np.ndarray, rotation_candidates: list[list[np.ndarray | None]]
) -> tuple[float, list[int], float, float | None]:
    """Pick the smoothest lambda whose selected-path mean score doesn't degrade more
    than IOU_DEGRADATION_TOLERANCE (relative) from the pure-argmax (lambda=0) choice.
    """
    baseline_path = dp_select_path(score_matrix, rotation_candidates, 0.0)
    baseline_score, baseline_jerk = path_metrics(baseline_path, score_matrix, rotation_candidates)
    min_acceptable_score = baseline_score * (1.0 - IOU_DEGRADATION_TOLERANCE)

    best_lambda, best_path, best_score, best_jerk = 0.0, baseline_path, baseline_score, baseline_jerk
    for lambda_weight in LAMBDA_GRID[1:]:
        path = dp_select_path(score_matrix, rotation_candidates, lambda_weight)
        mean_score, jerk = path_metrics(path, score_matrix, rotation_candidates)
        if mean_score < min_acceptable_score:
            continue
        if jerk is not None and (best_jerk is None or jerk < best_jerk):
            best_lambda, best_path, best_score, best_jerk = lambda_weight, path, mean_score, jerk

    return best_lambda, best_path, best_score, best_jerk


def normalize_focal_length(pose_data: dict, candidate_name: str) -> None:
    candidate_upscale = CANDIDATE_UPSCALE.get(candidate_name, CANONICAL_UPSCALE)
    # Recorded regardless of whether a rescale is needed: downstream stages (e.g.
    # render_final_poses.py's keypoints_2d fallback) need to know which pixel space
    # pred_keypoints_2d/pred_cam_t are actually in, since only focal_length is
    # renormalized here.
    pose_data["source_upscale"] = candidate_upscale
    if candidate_upscale == CANONICAL_UPSCALE:
        return
    pose_data["focal_length_raw"] = pose_data["focal_length"]
    pose_data["focal_length"] = pose_data["focal_length"] * (CANONICAL_UPSCALE / candidate_upscale)


def main():
    parser = argparse.ArgumentParser(
        description="Sequence-aware (Viterbi/DP) best-pose selection across multiple inference runs"
    )
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        default=None,
        metavar="NAME=DIR",
        help="name=inference_dir pair; repeatable. Default: the three pure per-strategy "
        "candidate dirs preserved before the previous argmax selection overwrote output/inference: "
        "hypir4x=output/inference_hypir4x_only hypir8x=output/inference_v2 lanczos4x=output/inference_lanczos4x",
    )
    parser.add_argument("--output-dir", default="output/inference_dp")
    parser.add_argument("--report", default="output/inference_dp/selection_report.json")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path

    if args.candidates:
        candidate_specs = args.candidates
    else:
        candidate_specs = [
            "hypir4x=output/inference_hypir4x_only",
            "hypir8x=output/inference_v2",
            "lanczos4x=output/inference_lanczos4x",
        ]

    candidate_names = []
    candidate_dirs: list[Path] = []
    for spec in candidate_specs:
        name, _, rel_dir = spec.partition("=")
        candidate_path = Path(rel_dir)
        if not candidate_path.is_absolute():
            candidate_path = REPO_ROOT / candidate_path
        candidate_names.append(name)
        candidate_dirs.append(candidate_path)

    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    selection_counts: dict[str, int] = {name: 0 for name in candidate_names}
    char_reports = []
    lambda_used = {}

    def process_group(
        char_id: int,
        sequence_name: str,
        angle_index: int,
        frame_infos: list[dict],
    ) -> dict | None:
        num_frames = len(frame_infos)
        num_candidates = len(candidate_names)
        score_matrix = np.full((num_frames, num_candidates), -np.inf)
        rotation_candidates: list[list[np.ndarray | None]] = [[None] * num_candidates for _ in range(num_frames)]
        frame_data_grid: list[list[dict | None]] = [[None] * num_candidates for _ in range(num_frames)]

        for t, frame_info in enumerate(frame_infos):
            frame_idx = inference_frame_dir_index(frame_info)
            frame_path = split_dir / frame_info["path"]
            for c, candidate_root in enumerate(candidate_dirs):
                data = load_candidate_frame_data(candidate_root, char_id, frame_idx, frame_path)
                if data is None:
                    continue
                frame_data_grid[t][c] = data
                score_matrix[t, c] = data["score"]
                rotation_candidates[t][c] = data["global_rots"]

        if not np.isfinite(score_matrix).any(axis=1).any():
            return None

        lambda_weight, path, selected_mean_score, selected_jerk = select_lambda_for_character(
            score_matrix, rotation_candidates
        )
        group_key = f"{char_id}:{sequence_name}:a{angle_index}"
        lambda_used[group_key] = lambda_weight
        print(
            f"char_{char_id:02d}/{sequence_name}/a{angle_index}: lambda={lambda_weight:g} "
            f"mean_score={selected_mean_score:.3f} "
            f"jerk={selected_jerk if selected_jerk is None else f'{selected_jerk:.2f}deg'}"
        )

        char_output_dir = output_dir / f"char_{char_id:02d}"
        frame_reports = []
        for t, frame_info in enumerate(frame_infos):
            frame_idx = inference_frame_dir_index(frame_info)
            selected_c = path[t]
            data = frame_data_grid[t][selected_c]
            if data is None:
                print(f"  WARNING: no candidate output for char_{char_id:02d} frame_{frame_idx:03d}")
                continue

            selected_name = candidate_names[selected_c]
            selection_counts[selected_name] += 1

            dest_dir = char_output_dir / f"frame_{frame_idx:03d}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for filename in CANDIDATE_FILES:
                source_file = data["frame_dir"] / filename
                if not source_file.exists():
                    continue
                if filename == "pose.json":
                    with open(source_file, encoding="utf-8") as pose_file:
                        pose_data = json.load(pose_file)
                    normalize_focal_length(pose_data, selected_name)
                    with open(dest_dir / filename, "w", encoding="utf-8") as pose_file:
                        json.dump(pose_data, pose_file, indent=2)
                else:
                    shutil.copy2(source_file, dest_dir / filename)

            frame_reports.append(
                {
                    "frame_index": frame_idx,
                    "sequence_name": sequence_name,
                    "angle_index": angle_index,
                    "selected": selected_name,
                    "scores": {
                        candidate_names[c]: (
                            float(score_matrix[t, c]) if np.isfinite(score_matrix[t, c]) else None
                        )
                        for c in range(num_candidates)
                    },
                }
            )

        return {
            "char_id": char_id,
            "sequence_name": sequence_name,
            "angle_index": angle_index,
            "lambda": lambda_weight,
            "mean_score": selected_mean_score,
            "temporal_jerk_deg": selected_jerk,
            "frames": frame_reports,
        }

    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        groups = group_frames_by_sequence_angle(char_entry)
        group_reports = []
        for (sequence_name, angle_index), frame_infos in sorted(groups.items()):
            report = process_group(char_id, sequence_name, angle_index, frame_infos)
            if report is not None:
                group_reports.append(report)

        if not group_reports:
            print(f"  WARNING: no candidate output at all for char_{char_id:02d}")
            continue

        mean_score = float(np.mean([group["mean_score"] for group in group_reports]))
        char_reports.append(
            {
                "char_id": char_id,
                "mean_score": mean_score,
                "groups": group_reports,
            }
        )

    print("\nSelection counts:")
    for name, count in selection_counts.items():
        print(f"  {name}: {count}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(
            {"selection_counts": selection_counts, "lambda_per_group": lambda_used, "characters": char_reports},
            report_file,
            indent=2,
        )
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
