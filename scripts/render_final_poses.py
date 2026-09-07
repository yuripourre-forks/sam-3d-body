#!/usr/bin/env python3
"""Stage 6: shared ground plane + shared camera/scale + bounded per-frame nudge.

Reads pose.json from --input-dir (the Stage 5 temporally-smoothed output) and,
per character:

  1. Computes one 3D ground offset from *all* of that character's frames
     (extending compute_character_ground_offset, previously only given the
     first 5 preview frames) and grounds pred_vertices/pred_keypoints_3d/
     pred_joint_coords with it -- this is the saved, BVH-consumed geometry.
  2. Computes one shared "floor row" in canvas pixel space from the sprite's
     own lowest content row across all of that character's frames.
  3. Renders every frame with the Stage 1 fitted shared camera (elevation/
     azimuth/roll) and shared per-character scale (extent_reference /
     scale_factor from output/camera.json) via render_mesh_shared_placed,
     instead of the old per-frame-rescaled frontal camera.
  4. Searches a small bounded per-frame nudge (translation, yaw, scale -- see
     NUDGE_*_OPTIONS) on top of that base render to maximize silhouette IoU
     against the sprite, implemented as cheap 2D affine warps of the already-
     rendered RGBA (not re-running pyrender per candidate).
  5. Temporally smooths the chosen nudge parameter sequence (Savitzky-Golay,
     matching Stage 5's window/order) so the correction can't reintroduce
     frame-to-frame jitter, then re-applies the smoothed nudge for the final
     mesh_render.png.

Writes pose.json/mesh.ply/mesh_render.png/mesh_iso.png to --output-dir
(output/inference_final by default) -- the final pipeline output Stage 7
reassembles, exports BVH from, and reports metrics for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.foot_grounding import (  # noqa: E402
    apply_ground_translation,
    compute_character_floor_row,
    compute_character_ground_offset,
    compute_frame_content_bbox_and_centroid,
    keypoints_2d_target,
    render_mesh_isometric,
    render_mesh_shared_placed,
    save_rgba_image,
    scale_content_target,
)
from scripts.manifest_utils import inference_frame_dir_index  # noqa: E402
from scripts.pose_metrics import composite_score, content_mask, mask_iou  # noqa: E402
from sam_3d_body.visualization.renderer import Renderer  # noqa: E402

RENDER_SCALE = 4
ISO_RENDER_SIZE = 512

# Bounded nudge search grid. Translation options are expressed as multiples of
# the *original* 96px sprite pixel (then scaled by RENDER_SCALE), matching the
# plan's "a few px" bound; yaw/scale bounds are similarly small/conservative.
NUDGE_PX_OPTIONS = [-2, -1, 0, 1, 2]
NUDGE_YAW_OPTIONS_DEG = [-4.0, 0.0, 4.0]
NUDGE_SCALE_OPTIONS = [0.95, 1.0, 1.05]

SAVGOL_WINDOW = 5
SAVGOL_ORDER = 2


def load_manifest(split_dir: Path) -> dict:
    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def load_faces(faces_source_dir: Path, manifest: dict) -> np.ndarray:
    for char_entry in manifest["characters"]:
        for frame_info in char_entry["frames"]:
            mesh_path = faces_source_dir / f"char_{char_entry['id']:02d}" / f"frame_{inference_frame_dir_index(frame_info):03d}" / "mesh.ply"
            if mesh_path.exists():
                return trimesh.load(str(mesh_path), process=False).faces
    raise FileNotFoundError("No mesh.ply found to read face topology from")


BOUNDARY_LAMBDA = 0.002


def load_character_frames(input_dir: Path, char_entry: dict) -> list[dict]:
    frame_lookup = {frame_info["index"]: frame_info for frame_info in char_entry["frames"]}
    frames = []
    for frame_info in char_entry["frames"]:
        pose_path = input_dir / f"char_{char_entry['id']:02d}" / f"frame_{inference_frame_dir_index(frame_info):03d}" / "pose.json"
        if not pose_path.exists():
            continue
        with open(pose_path, encoding="utf-8") as pose_file:
            pose_data = json.load(pose_file)
        pose_data["_frame_index"] = frame_info["index"]
        pose_data["_local_index"] = inference_frame_dir_index(frame_info)
        pose_data["_frame_path"] = frame_info["path"]
        pose_data["_source_bbox"] = frame_info.get("source_bbox")
        pose_data["_view_yaw_deg"] = float(frame_info.get("view_yaw_deg", 0.0))
        pose_data["_sequence_name"] = frame_info.get("sequence_name", f"char_{char_entry['id']:02d}")
        pose_data["_angle_index"] = int(frame_info.get("angle_index", 0))
        frames.append(pose_data)
    return frames


def apply_nudge(rgba_uint8: np.ndarray, dx: float, dy: float, yaw_deg: float, scale: float) -> np.ndarray:
    height, width = rgba_uint8.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, yaw_deg, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(
        rgba_uint8, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )


def search_best_nudge(
    base_rgba_uint8: np.ndarray, sprite_mask: np.ndarray, render_scale: int
) -> tuple[tuple[float, float, float, float], float]:
    best_params = (0.0, 0.0, 0.0, 1.0)
    base_mask = base_rgba_uint8[:, :, 3] > 0
    best_iou = mask_iou(sprite_mask, base_mask) or 0.0

    for dx_px in NUDGE_PX_OPTIONS:
        for dy_px in NUDGE_PX_OPTIONS:
            for yaw in NUDGE_YAW_OPTIONS_DEG:
                for scale in NUDGE_SCALE_OPTIONS:
                    if dx_px == 0 and dy_px == 0 and yaw == 0.0 and scale == 1.0:
                        continue
                    dx, dy = dx_px * render_scale, dy_px * render_scale
                    warped = apply_nudge(base_rgba_uint8, dx, dy, yaw, scale)
                    warped_mask = warped[:, :, 3] > 0
                    score = composite_score(sprite_mask, warped_mask, boundary_lambda=BOUNDARY_LAMBDA)
                    if score is not None and score > best_iou:
                        best_iou = score
                        best_params = (float(dx), float(dy), float(yaw), float(scale))
    return best_params, best_iou


def smooth_nudge_sequence(nudge_params: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    matrix = np.array(nudge_params, dtype=np.float32)
    window = min(SAVGOL_WINDOW, len(matrix) if len(matrix) % 2 == 1 else len(matrix) - 1)
    if window < SAVGOL_ORDER + 2:
        return nudge_params
    smoothed = savgol_filter(matrix, window_length=window, polyorder=SAVGOL_ORDER, mode="interp", axis=0)
    return [tuple(row) for row in smoothed]


def sprite_frame_target(
    split_dir: Path, frame_path: str, render_scale: int, keypoints_2d: np.ndarray, upscale_factor: int
) -> tuple[float, float]:
    """Per-frame target x-centroid in canvas pixel space (see module docstring)."""
    content = compute_frame_content_bbox_and_centroid(str(split_dir / frame_path))
    if content is not None:
        target = scale_content_target(content, render_scale)
        return target.centroid
    target = keypoints_2d_target(keypoints_2d * (render_scale / upscale_factor))
    return target.centroid


def save_mesh_ply(renderer: Renderer, vertices: np.ndarray, cam_t: np.ndarray, output_path: Path) -> None:
    mesh = renderer.vertices_to_trimesh(vertices, cam_t, (0.65, 0.74, 0.86))
    mesh.export(str(output_path))


def numpy_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared ground plane + shared camera + bounded nudge (Stage 6)")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--input-dir", default="output/inference_smoothed")
    parser.add_argument("--faces-source-dir", default="output/inference")
    parser.add_argument("--camera", default="output/camera.json")
    parser.add_argument("--output-dir", default="output/inference_final")
    parser.add_argument("--render-scale", type=int, default=RENDER_SCALE)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    faces_source_dir = Path(args.faces_source_dir)
    if not faces_source_dir.is_absolute():
        faces_source_dir = REPO_ROOT / faces_source_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    camera_path = Path(args.camera)
    if not camera_path.is_absolute():
        camera_path = REPO_ROOT / camera_path

    with open(camera_path, encoding="utf-8") as camera_file:
        camera = json.load(camera_file)

    manifest = load_manifest(split_dir)
    faces = load_faces(faces_source_dir, manifest)

    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        frames = load_character_frames(input_dir, char_entry)
        if not frames:
            continue

        char_scale_factor = camera["per_character_scale_factor"][str(char_id)]
        extent_override = camera["extent_reference"][str(char_id)] / char_scale_factor

        # Stage 6.1: one 3D ground offset from every frame of this character.
        ground_offset = compute_character_ground_offset(
            [{"pred_keypoints_3d": np.array(frame["pred_keypoints_3d"], dtype=np.float32)} for frame in frames]
        )

        # Stage 6.2: one shared floor row (canvas pixel space) from the sprite's own frames.
        output_frame_size = manifest.get("output_frame_size", manifest.get("frame_size", 96))
        canvas_width = output_frame_size * args.render_scale
        canvas_height = output_frame_size * args.render_scale
        per_angle_nudge = camera.get("per_angle_nudge_deg", {})
        sprite_masks = []
        for frame in frames:
            rgba = np.array(Image.open(split_dir / frame["_frame_path"]).convert("RGBA"))
            resized = np.array(
                Image.fromarray(rgba).resize((canvas_width, canvas_height), Image.NEAREST)
            )
            sprite_masks.append(content_mask(resized))
        floor_row = compute_character_floor_row(sprite_masks)

        char_output_dir = output_dir / f"char_{char_id:02d}"
        base_renders = []
        nudge_choices = []
        for frame, sprite_mask in zip(frames, sprite_masks):
            vertices = np.array(frame["pred_vertices"], dtype=np.float32)
            keypoints_3d = np.array(frame["pred_keypoints_3d"], dtype=np.float32)
            joint_coords = np.array(frame["pred_joint_coords"], dtype=np.float32)
            grounded_vertices, grounded_keypoints_3d, grounded_joints = apply_ground_translation(
                vertices, keypoints_3d, joint_coords=joint_coords, extra_offset=ground_offset
            )

            target_centroid_x, _ = sprite_frame_target(
                split_dir,
                frame["_frame_path"],
                args.render_scale,
                np.array(frame["pred_keypoints_2d"], dtype=np.float32),
                upscale_factor=frame.get("source_upscale", 4),
            )

            angle_nudge = (
                per_angle_nudge.get(str(char_id), {})
                .get(frame["_sequence_name"], {})
                .get(str(frame["_angle_index"]), 0.0)
            )
            base_rgba = render_mesh_shared_placed(
                grounded_vertices,
                faces,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                elevation_deg=camera["elevation_deg"],
                azimuth_deg=camera["azimuth_deg"],
                roll_deg=camera["roll_deg"],
                extent_override=extent_override,
                target_centroid_x=target_centroid_x,
                target_floor_row=floor_row,
                view_yaw_deg=frame["_view_yaw_deg"],
                angle_nudge_deg=float(angle_nudge),
            )
            base_uint8 = np.clip(base_rgba * 255, 0, 255).astype(np.uint8) if base_rgba.dtype != np.uint8 else base_rgba
            base_renders.append((base_uint8, grounded_vertices, grounded_keypoints_3d, grounded_joints))

            nudge_params, _ = search_best_nudge(base_uint8, sprite_mask, args.render_scale)
            nudge_choices.append(nudge_params)

        smoothed_nudges = smooth_nudge_sequence(nudge_choices)

        char_output_dir.mkdir(parents=True, exist_ok=True)
        for frame, (base_uint8, grounded_vertices, grounded_keypoints_3d, grounded_joints), nudge, sprite_mask in zip(
            frames, base_renders, smoothed_nudges, sprite_masks
        ):
            frame_idx = frame["_local_index"]
            final_rgba = apply_nudge(base_uint8, *nudge)
            final_iou = mask_iou(sprite_mask, final_rgba[:, :, 3] > 0)

            renderer = Renderer(focal_length=frame["focal_length"], faces=faces)
            cam_t = np.array(frame["pred_cam_t"], dtype=np.float32)

            frame_out_dir = char_output_dir / f"frame_{frame_idx:03d}"
            frame_out_dir.mkdir(parents=True, exist_ok=True)

            save_rgba_image(final_rgba, str(frame_out_dir / "mesh_render.png"))
            save_mesh_ply(renderer, grounded_vertices, cam_t, frame_out_dir / "mesh.ply")

            angle_nudge = (
                per_angle_nudge.get(str(char_id), {})
                .get(frame["_sequence_name"], {})
                .get(str(frame["_angle_index"]), 0.0)
            )
            iso_rgba = render_mesh_isometric(
                grounded_vertices,
                faces,
                render_size=ISO_RENDER_SIZE,
                elevation_deg=camera["elevation_deg"],
                azimuth_deg=camera["azimuth_deg"],
                roll_deg=camera["roll_deg"],
                extent_override=extent_override,
                view_yaw_deg=frame["_view_yaw_deg"] + float(angle_nudge),
            )
            save_rgba_image(iso_rgba, str(frame_out_dir / "mesh_iso.png"))

            new_pose_data = {key: value for key, value in frame.items() if not key.startswith("_")}
            new_pose_data["pred_vertices"] = numpy_to_jsonable(grounded_vertices)
            new_pose_data["pred_keypoints_3d"] = numpy_to_jsonable(grounded_keypoints_3d)
            new_pose_data["pred_joint_coords"] = numpy_to_jsonable(grounded_joints)
            new_pose_data["ground_offset"] = float(ground_offset)
            new_pose_data["nudge_dx_dy_yaw_scale"] = [float(v) for v in nudge]
            new_pose_data["final_iou"] = final_iou
            new_pose_data["frame_index"] = frame_idx
            new_pose_data["char_id"] = char_id
            with open(frame_out_dir / "pose.json", "w", encoding="utf-8") as pose_file:
                json.dump(new_pose_data, pose_file, indent=2)

        print(f"char_{char_id:02d}: rendered {len(frames)} frames (ground_offset={ground_offset:.4f}, floor_row={floor_row:.1f})")

    print(f"Done. Wrote: {output_dir}")


if __name__ == "__main__":
    main()
