#!/usr/bin/env python3
"""Minimal single-image sprite pipeline: slice, upscale, SAM3D-Body, overlay, BVH."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.visualization.renderer import Renderer
from scripts.export_bvh import DEFAULT_FPS, write_bvh
from scripts.foot_grounding import render_mesh_overlay, save_rgba_image
from scripts.hypir_upscale import (
    DEFAULT_BASE_MODEL,
    DEFAULT_WEIGHT_PATH,
    HypirUpscaler,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/model.ckpt"
DEFAULT_MHR_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
DEFAULT_UPSCALE_FACTOR = 4
NEUTRAL_BG_BGR = (128, 128, 128)
FRAME_NAME_WIDTH = 3


def numpy_to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_rect(rect_str: str) -> tuple[int, int, int, int]:
    """Parse rect as x,y,width,height."""
    parts = [int(part.strip()) for part in rect_str.split(",")]
    if len(parts) != 4:
        raise ValueError("rect must be four comma-separated integers: x,y,width,height")
    x, y, width, height = parts
    if width <= 0 or height <= 0:
        raise ValueError("rect width and height must be positive")
    return x, y, width, height


def load_and_crop_rgba(image_path: Path, rect: tuple[int, int, int, int] | None) -> np.ndarray:
    image = Image.open(image_path).convert("RGBA")
    if rect is not None:
        x, y, width, height = rect
        image = image.crop((x, y, x + width, y + height))
    return np.array(image)


def slice_horizontal_frames(rgba: np.ndarray, num_frames: int) -> list[np.ndarray]:
    if num_frames <= 0:
        raise ValueError("frames must be a positive integer")
    sheet_height, sheet_width = rgba.shape[:2]
    column_width = sheet_width // num_frames
    if column_width <= 0:
        raise ValueError(
            f"Image width {sheet_width} is too small to slice into {num_frames} frames"
        )
    frames = []
    for frame_index in range(num_frames):
        x0 = frame_index * column_width
        x1 = x0 + column_width if frame_index < num_frames - 1 else sheet_width
        frames.append(rgba[:, x0:x1].copy())
    return frames


def rgba_to_bgr_on_neutral(rgba: np.ndarray) -> np.ndarray:
    """Composite RGBA onto a neutral gray background."""
    bgr = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3]
    neutral = np.full_like(bgr, NEUTRAL_BG_BGR, dtype=np.uint8)
    alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
    blended = bgr.astype(np.float32) * alpha_f + neutral.astype(np.float32) * (1.0 - alpha_f)
    return blended.astype(np.uint8)


def save_bgr_image(image_bgr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)


def save_pose_json(output: dict, output_path: Path) -> None:
    pose_data = {
        "pred_keypoints_3d": numpy_to_jsonable(output["pred_keypoints_3d"]),
        "pred_keypoints_2d": numpy_to_jsonable(output["pred_keypoints_2d"]),
        "pred_vertices": numpy_to_jsonable(output["pred_vertices"]),
        "pred_cam_t": numpy_to_jsonable(output["pred_cam_t"]),
        "pred_global_rots": numpy_to_jsonable(output["pred_global_rots"]),
        "focal_length": numpy_to_jsonable(output["focal_length"]),
        "bbox": numpy_to_jsonable(output["bbox"]),
        "frame_index": output.get("frame_index"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as pose_file:
        json.dump(pose_data, pose_file, indent=2)


def render_mesh_transparent(
    renderer: Renderer,
    vertices: np.ndarray,
    cam_t: np.ndarray,
    background_bgr: np.ndarray,
) -> np.ndarray:
    """Render mesh-only RGBA with transparent background."""
    rgba = renderer(
        vertices,
        cam_t,
        background_bgr,
        scene_bg_color=(0, 0, 0),
        return_rgba=True,
    )
    return rgba


def frame_name(frame_index: int) -> str:
    return f"frame_{frame_index:0{FRAME_NAME_WIDTH}d}"


def run_inference_on_frame(
    estimator: SAM3DBodyEstimator,
    upscaled_bgr: np.ndarray,
) -> dict | None:
    height, width = upscaled_bgr.shape[:2]
    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    bbox = np.array([[0, 0, width, height]], dtype=np.float32)
    outputs = estimator.process_one_image(
        upscaled_rgb,
        bboxes=bbox,
        inference_type="body",
    )
    if not outputs:
        return None
    return outputs[0]


def process_frame(
    frame_index: int,
    frame_rgba: np.ndarray,
    output_dir: Path,
    estimator: SAM3DBodyEstimator,
    use_upscale: bool,
    upscale_factor: int,
    hypir_upscaler: HypirUpscaler | None,
) -> dict | None:
    frame_bgr = rgba_to_bgr_on_neutral(frame_rgba)
    frame_stem = frame_name(frame_index)

    frames_dir = output_dir / "frames"
    upscaled_dir = output_dir / "upscaled"
    inference_dir = output_dir / "inference" / frame_stem

    save_bgr_image(frame_bgr, frames_dir / f"{frame_stem}.png")

    if use_upscale and hypir_upscaler is not None:
        upscaled_bgr = hypir_upscaler.upscale_bgr(frame_bgr)
    else:
        upscaled_bgr = frame_bgr.copy()
    save_bgr_image(upscaled_bgr, upscaled_dir / f"{frame_stem}.png")

    output = run_inference_on_frame(estimator, upscaled_bgr)
    if output is None:
        print(f"WARNING: no SAM3D output for {frame_stem}")
        return None

    output["frame_index"] = frame_index
    inference_dir.mkdir(parents=True, exist_ok=True)

    renderer = Renderer(focal_length=output["focal_length"], faces=estimator.faces)
    vertices = output["pred_vertices"]
    cam_t = output["pred_cam_t"]

    mesh_rgba = render_mesh_transparent(renderer, vertices, cam_t, upscaled_bgr)
    save_rgba_image(mesh_rgba, str(inference_dir / "mesh.png"))

    overlay_bgr = render_mesh_overlay(renderer, vertices, cam_t, upscaled_bgr)
    save_bgr_image(overlay_bgr, inference_dir / "overlay.png")

    save_pose_json(output, inference_dir / "pose.json")
    print(f"  {frame_stem} done")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slice one image into frames and run upscale -> SAM3D-Body -> overlay -> BVH"
    )
    parser.add_argument("--image", required=True, help="Input sprite sheet image path")
    parser.add_argument("--frames", type=int, required=True, help="Number of horizontal frames")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--rect",
        default=None,
        help="Optional crop before slicing, as x,y,width,height",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale each frame with HYPIR before inference",
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=DEFAULT_UPSCALE_FACTOR,
        help="HYPIR upscale factor when --upscale is set",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mhr-path", default=str(DEFAULT_MHR_PATH))
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (default: cuda if available else cpu)",
    )
    parser.add_argument("--hypir-weight", default=str(DEFAULT_WEIGHT_PATH))
    parser.add_argument("--hypir-base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="BVH playback FPS")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = resolve_path(args.image)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rect = parse_rect(args.rect) if args.rect is not None else None
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rgba = load_and_crop_rgba(image_path, rect)
    frame_rgbs = slice_horizontal_frames(rgba, args.frames)
    print(f"Sliced {len(frame_rgbs)} frames from {image_path}")

    hypir_upscaler = None
    if args.upscale:
        hypir_upscaler = HypirUpscaler(
            weight_path=args.hypir_weight,
            base_model_path=args.hypir_base_model,
            device=device,
            upscale=args.upscale_factor,
        )
        hypir_upscaler.load()

    print("Loading SAM3D-Body model...")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=args.checkpoint,
        mhr_path=args.mhr_path,
        device=device,
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    frame_outputs: list[dict] = []
    for frame_index, frame_rgba in enumerate(frame_rgbs):
        output = process_frame(
            frame_index=frame_index,
            frame_rgba=frame_rgba,
            output_dir=output_dir,
            estimator=estimator,
            use_upscale=args.upscale,
            upscale_factor=args.upscale_factor,
            hypir_upscaler=hypir_upscaler,
        )
        if output is not None:
            frame_outputs.append(output)

    if not frame_outputs:
        raise RuntimeError("No frames produced SAM3D output; animation.bvh was not written")

    bvh_path = output_dir / "animation.bvh"
    write_bvh(frame_outputs, bvh_path, fps=args.fps)
    print(f"Exported BVH: {bvh_path} ({len(frame_outputs)} frames)")
    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
