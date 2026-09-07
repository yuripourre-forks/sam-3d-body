#!/usr/bin/env python3
"""Run SAM3D-Body inference on split townsfolk sprite frames."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.visualization.renderer import Renderer
from scripts.foot_grounding import (
    apply_ground_translation,
    assemble_sprite_sheet,
    compute_character_ground_offset,
    compute_frame_content_bbox_and_centroid,
    render_mesh_frontal_placed,
    render_mesh_isometric,
    save_rgba_image,
    scale_content_target,
)
from scripts.hypir_upscale import DEFAULT_WEIGHT_PATH, HypirUpscaler

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/model.ckpt"
DEFAULT_MHR_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
UPSCALE_FACTOR = 8
# Scale used for mesh_render.png canvases / reassembly slot placement. Kept
# independent of --upscale (which only controls the resolution fed into
# SAM3D-Body) so raising --upscale for better pose quality doesn't blow up
# render canvas sizes or break scripts/reassemble_mesh_render.py's hardcoded
# UPSCALE_FACTOR=4 slot placement math.
RENDER_SCALE = 4
DEFAULT_RENDER_SIZE = 512
NEUTRAL_BG_BGR = (128, 128, 128)
MAGENTA_THRESHOLD_BGR = (200, 100, 200)


def numpy_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def save_pose_json(output: dict, output_path: Path) -> None:
    pose_data = {
        "pred_keypoints_3d": numpy_to_jsonable(output["pred_keypoints_3d"]),
        "pred_keypoints_2d": numpy_to_jsonable(output["pred_keypoints_2d"]),
        "pred_vertices": numpy_to_jsonable(output["pred_vertices"]),
        "pred_cam_t": numpy_to_jsonable(output["pred_cam_t"]),
        "pred_pose_raw": numpy_to_jsonable(output["pred_pose_raw"]),
        "global_rot": numpy_to_jsonable(output["global_rot"]),
        "body_pose_params": numpy_to_jsonable(output["body_pose_params"]),
        "hand_pose_params": numpy_to_jsonable(output["hand_pose_params"]),
        "scale_params": numpy_to_jsonable(output["scale_params"]),
        "shape_params": numpy_to_jsonable(output["shape_params"]),
        "expr_params": numpy_to_jsonable(output["expr_params"]),
        "pred_joint_coords": numpy_to_jsonable(output["pred_joint_coords"]),
        "pred_global_rots": numpy_to_jsonable(output["pred_global_rots"]),
        "mhr_model_params": numpy_to_jsonable(output["mhr_model_params"]),
        "focal_length": numpy_to_jsonable(output["focal_length"]),
        "bbox": numpy_to_jsonable(output["bbox"]),
        "ground_offset": numpy_to_jsonable(output.get("ground_offset", 0.0)),
        "frame_index": output.get("frame_index"),
        "char_id": output.get("char_id"),
        "source_frame": output.get("source_frame"),
    }
    with open(output_path, "w", encoding="utf-8") as pose_file:
        json.dump(pose_data, pose_file, indent=2)


def load_frame_bgr(frame_path: Path) -> np.ndarray:
    """Load frame and composite onto neutral background."""
    rgba = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise ValueError(f"Could not load frame: {frame_path}")

    if rgba.shape[2] == 4:
        bgr = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3]
        neutral = np.full_like(bgr, NEUTRAL_BG_BGR, dtype=np.uint8)
        alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
        bgr = (bgr.astype(np.float32) * alpha_f + neutral.astype(np.float32) * (1 - alpha_f))
        bgr = bgr.astype(np.uint8)
    else:
        bgr = rgba

    magenta_mask = (
        (bgr[:, :, 2] > MAGENTA_THRESHOLD_BGR[0])
        & (bgr[:, :, 1] < MAGENTA_THRESHOLD_BGR[1])
        & (bgr[:, :, 0] > MAGENTA_THRESHOLD_BGR[2])
    )
    bgr[magenta_mask] = NEUTRAL_BG_BGR
    return bgr


def upscale_frame_bgr(
    bgr: np.ndarray,
    upscale_factor: int,
    hypir_upscaler: HypirUpscaler | None,
) -> np.ndarray:
    if hypir_upscaler is not None:
        return hypir_upscaler.upscale_bgr(bgr)

    height, width = bgr.shape[:2]
    return cv2.resize(
        bgr,
        (width * upscale_factor, height * upscale_factor),
        interpolation=cv2.INTER_LANCZOS4,
    )


def preprocess_frame(
    frame_path: Path,
    upscale_factor: int,
    hypir_upscaler: HypirUpscaler | None = None,
    save_upscaled_path: Path | None = None,
    pre_upscaled_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load frame, remove magenta background, upscale for inference.

    If `pre_upscaled_path` points to an existing file (produced ahead of time by
    scripts/upscale_townsfolk.py), it is loaded directly instead of re-running the
    upscaler inline -- this decouples the (slow, model-loading) upscale step from
    inference so it only needs to run once per frame regardless of how many times
    process_townsfolk.py is re-run.
    """
    bgr = load_frame_bgr(frame_path)

    if pre_upscaled_path is not None and pre_upscaled_path.exists():
        upscaled = cv2.imread(str(pre_upscaled_path), cv2.IMREAD_COLOR)
        if upscaled is None:
            raise ValueError(f"Could not load pre-upscaled frame: {pre_upscaled_path}")
    else:
        upscaled = upscale_frame_bgr(bgr, upscale_factor, hypir_upscaler)
        if save_upscaled_path is not None:
            save_upscaled_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_upscaled_path), upscaled)

    return bgr, upscaled


def run_inference(
    estimator,
    frame_path: Path,
    upscale_factor: int,
    inference_type: str,
    hypir_upscaler: HypirUpscaler | None = None,
    save_upscaled_path: Path | None = None,
    pre_upscaled_path: Path | None = None,
) -> dict | None:
    _, upscaled_bgr = preprocess_frame(
        frame_path,
        upscale_factor,
        hypir_upscaler=hypir_upscaler,
        save_upscaled_path=save_upscaled_path,
        pre_upscaled_path=pre_upscaled_path,
    )
    height, width = upscaled_bgr.shape[:2]
    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    bbox = np.array([[0, 0, width, height]], dtype=np.float32)
    outputs = estimator.process_one_image(
        upscaled_rgb,
        bboxes=bbox,
        inference_type=inference_type,
    )
    if not outputs:
        return None
    return outputs[0]


def save_mesh_ply(
    renderer: Renderer,
    vertices: np.ndarray,
    cam_t: np.ndarray,
    output_path: Path,
) -> None:
    mesh = renderer.vertices_to_trimesh(vertices, cam_t, (0.65, 0.74, 0.86))
    mesh.export(str(output_path))


def load_manifest(split_dir: Path) -> dict:
    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def main():
    parser = argparse.ArgumentParser(description="Process townsfolk frames with SAM3D-Body")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--output-dir", default="output/inference")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mhr-path", default=str(DEFAULT_MHR_PATH))
    parser.add_argument("--char-start", type=int, default=0)
    parser.add_argument("--char-end", type=int, default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--upscale", type=int, default=UPSCALE_FACTOR)
    parser.add_argument(
        "--render-scale",
        type=int,
        default=RENDER_SCALE,
        help="Scale for mesh_render.png canvases / reassembly slots (independent of --upscale)",
    )
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE)
    parser.add_argument("--inference-type", default="body", choices=["body", "full"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-hypir", action="store_true", default=True)
    parser.add_argument("--no-hypir", action="store_false", dest="use_hypir")
    parser.add_argument(
        "--hypir-base-model",
        default=str(REPO_ROOT / "checkpoints/sd2-1-base"),
        help="Path to SD2.1 base model (local dir or HF repo id)",
    )
    parser.add_argument("--hypir-weight", default=str(DEFAULT_WEIGHT_PATH))
    parser.add_argument("--hypir-prompt", default="isometric pixel art game character sprite, sharp details")
    parser.add_argument("--hypir-patch-size", type=int, default=512)
    parser.add_argument("--hypir-stride", type=int, default=256)
    parser.add_argument("--save-upscaled", action="store_true", help="Save upscaled frames")
    parser.add_argument(
        "--pre-upscaled-dir",
        default=None,
        help="Directory of frames already upscaled by scripts/upscale_townsfolk.py "
        "(char_XX/frame_YYY.png layout matching --split-dir). When a frame exists "
        "here it is used directly instead of upscaling inline.",
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    upscaled_dir = output_dir.parent / "upscaled"
    manifest = load_manifest(split_dir)
    char_end = args.char_end if args.char_end is not None else len(manifest["characters"])

    pre_upscaled_dir = None
    if args.pre_upscaled_dir is not None:
        pre_upscaled_dir = Path(args.pre_upscaled_dir)
        if not pre_upscaled_dir.is_absolute():
            pre_upscaled_dir = REPO_ROOT / pre_upscaled_dir

    # Lazily constructed: skipped entirely when every frame is already covered by
    # --pre-upscaled-dir, so re-running inference doesn't reload a diffusion model.
    hypir_upscaler_holder: list[HypirUpscaler] = []

    def get_hypir_upscaler() -> HypirUpscaler:
        if not hypir_upscaler_holder:
            upscaler = HypirUpscaler(
                weight_path=args.hypir_weight,
                base_model_path=args.hypir_base_model,
                device=args.device,
                upscale=args.upscale,
                prompt=args.hypir_prompt,
                patch_size=args.hypir_patch_size,
                stride=args.hypir_stride,
            )
            upscaler.load()
            hypir_upscaler_holder.append(upscaler)
        return hypir_upscaler_holder[0]

    print("Loading SAM3D-Body model...")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=args.checkpoint,
        mhr_path=args.mhr_path,
        device=args.device,
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    faces = estimator.faces

    def resolve_pre_upscaled_path(frame_info: dict) -> Path | None:
        if pre_upscaled_dir is None:
            return None
        candidate = pre_upscaled_dir / frame_info["path"]
        return candidate if candidate.exists() else None

    def run_frame_inference(frame_info: dict, frame_path: Path, char_id: int, frame_idx: int):
        pre_upscaled_path = resolve_pre_upscaled_path(frame_info)
        save_path = None
        if args.save_upscaled and pre_upscaled_path is None:
            save_path = upscaled_dir / f"char_{char_id:02d}" / f"frame_{frame_idx:03d}.png"
        upscaler = None
        if pre_upscaled_path is None and args.use_hypir:
            upscaler = get_hypir_upscaler()
        return run_inference(
            estimator,
            frame_path,
            args.upscale,
            args.inference_type,
            hypir_upscaler=upscaler,
            save_upscaled_path=save_path,
            pre_upscaled_path=pre_upscaled_path,
        )

    all_character_outputs = {}
    for char_entry in manifest["characters"][args.char_start:char_end]:
        char_id = char_entry["id"]
        frame_infos = char_entry["frames"]
        if args.frame_end is not None:
            frame_infos = frame_infos[args.frame_start:args.frame_end]
        else:
            frame_infos = frame_infos[args.frame_start:]
        frame_paths = [split_dir / frame_info["path"] for frame_info in frame_infos]

        print(f"Processing char_{char_id:02d} ({len(frame_paths)} frames)...")

        preview_outputs = []
        preview_count = min(5, len(frame_paths))
        for frame_idx in range(preview_count):
            output = run_frame_inference(
                frame_infos[frame_idx], frame_paths[frame_idx], char_id, frame_idx
            )
            if output:
                preview_outputs.append(output)

        ground_offset = (
            compute_character_ground_offset(preview_outputs) if preview_outputs else 0.0
        )
        print(f"  ground offset: {ground_offset:.4f}")

        char_outputs = []
        iso_renders = []
        char_output_dir = output_dir / f"char_{char_id:02d}"
        char_output_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, frame_path in enumerate(frame_paths):
            if frame_idx < preview_count and frame_idx < len(preview_outputs):
                output = preview_outputs[frame_idx]
            else:
                output = run_frame_inference(
                    frame_infos[frame_idx], frame_path, char_id, frame_idx
                )
            if output is None:
                print(f"  WARNING: no output for char_{char_id:02d} frame_{frame_idx:03d}")
                continue

            frame_out_dir = char_output_dir / f"frame_{frame_idx:03d}"
            frame_out_dir.mkdir(parents=True, exist_ok=True)

            renderer = Renderer(focal_length=output["focal_length"], faces=faces)
            # cam_t is intentionally passed through unmodified: it is combined with
            # grounded_vertices via a single addition at render time, so the ground
            # offset must live in exactly one of the two operands (see
            # apply_ground_translation's docstring).
            grounded_vertices, grounded_keypoints_3d, grounded_joints = apply_ground_translation(
                output["pred_vertices"],
                output["pred_keypoints_3d"],
                joint_coords=output["pred_joint_coords"],
                extra_offset=ground_offset,
            )
            cam_t = output["pred_cam_t"]

            grounded_output = dict(output)
            grounded_output["pred_vertices"] = grounded_vertices
            grounded_output["pred_keypoints_3d"] = grounded_keypoints_3d
            grounded_output["pred_joint_coords"] = grounded_joints
            grounded_output["ground_offset"] = ground_offset
            grounded_output["frame_index"] = frame_idx
            grounded_output["char_id"] = char_id
            grounded_output["source_frame"] = str(frame_path)

            save_pose_json(grounded_output, frame_out_dir / "pose.json")
            save_mesh_ply(renderer, grounded_vertices, cam_t, frame_out_dir / "mesh.ply")

            # canvas_width/canvas_height must match the coordinate space that
            # `content` (below) is measured in: compute_frame_content_bbox_and_centroid
            # reads pixels directly from `frame_path`, which is the on-disk *normalized*
            # square split frame (scripts/sprite_grid.normalize_to_square), not the
            # frame's raw pre-normalization footprint. frame_info["source_bbox"] is
            # that pre-normalization content bbox *local to the frame's own grid cell*
            # (not full-sheet coordinates, and not the same size as the normalized
            # square), so it must never be used to size this canvas -- doing so mixes
            # two different coordinate spaces and pushes the mesh mostly/entirely off
            # canvas. Reassembling this render back onto the original sheet's own
            # (possibly non-square) cell footprint happens later, in
            # scripts/reassemble_mesh_render.py, via
            # sprite_grid.place_normalized_render_on_sheet().
            output_frame_size = manifest.get("output_frame_size", manifest.get("frame_size", 96))
            canvas_width = output_frame_size * args.render_scale
            canvas_height = output_frame_size * args.render_scale
            content = compute_frame_content_bbox_and_centroid(str(frame_path))
            target = scale_content_target(content, args.render_scale) if content is not None else None
            # Fallback path only (used when `target` is None): pred_keypoints_2d is in
            # SAM3D-Body's input pixel space (scaled by args.upscale), so it must be
            # rescaled into the render canvas's own coordinate space (args.render_scale)
            # before use, since the two scales are no longer required to match.
            keypoints_2d_in_render_space = output["pred_keypoints_2d"] * (
                args.render_scale / args.upscale
            )
            mesh_rgba = render_mesh_frontal_placed(
                grounded_vertices,
                faces,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                target=target,
                keypoints_2d=keypoints_2d_in_render_space,
            )
            save_rgba_image(mesh_rgba, str(frame_out_dir / "mesh_render.png"))

            iso_rgba = render_mesh_isometric(
                grounded_vertices,
                faces,
                render_size=args.render_size,
            )
            save_rgba_image(iso_rgba, str(frame_out_dir / "mesh_iso.png"))
            iso_renders.append(iso_rgba)

            char_outputs.append(grounded_output)
            print(f"  char_{char_id:02d} frame_{frame_idx:03d} done")

        sprites_dir = output_dir.parent / "sprites"
        sprites_dir.mkdir(parents=True, exist_ok=True)
        if iso_renders:
            assemble_sprite_sheet(
                iso_renders,
                str(sprites_dir / f"char_{char_id:02d}_sheet.png"),
            )

        all_character_outputs[char_id] = char_outputs

    summary_path = output_dir / "processing_summary.json"
    summary = {char_id: len(frames) for char_id, frames in all_character_outputs.items()}
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
