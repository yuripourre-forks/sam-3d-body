#!/usr/bin/env python3
"""Minimal single-image sprite pipeline: slice, upscale, SAM3D-Body, overlay, BVH.

Example (townsfolk walk cycle, 25 frames, crop height 95 to skip separator row):

    python scripts/basic_pipeline.py \\
        --image input/townsfolk.png \\
        --rect 0,0,2400,95 \\
        --frames 25 \\
        --upscale \\
        --loop \\
        --output output/townsfolk_stable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.spatial.transform as transform
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.exporters.bvh_exporter import BVHExporter
from sam_3d_body.visualization.renderer import Renderer
from scripts.export_bvh import DEFAULT_FPS, METERS_TO_CENTIMETERS, build_exporter, write_bvh
from scripts.foot_grounding import save_rgba_image
from scripts.hypir_upscale import (
    DEFAULT_BASE_MODEL,
    DEFAULT_WEIGHT_PATH,
    HypirUpscaler,
)
from scripts.mhr_repose import CAMERA_AXIS_FLIP, repose
from scripts.pose_priors import euler_zyx_to_rotmat, weighted_median_per_dim
from scripts.sprite_grid import content_mask, tight_content_bbox
from scripts.temporal_smooth_poses import (
    consecutive_distances,
    find_outlier_runs,
    find_outliers,
    repair_outliers,
    smooth_character,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/model.ckpt"
DEFAULT_MHR_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
DEFAULT_UPSCALE_FACTOR = 4
DEFAULT_TOWNSFOLK_RECT = "0,0,2400,95"
NEUTRAL_BG_RGB = (128, 128, 128)
MAGENTA_THRESHOLD_RGB = (200, 100, 200)
FRAME_NAME_WIDTH = 3
DEFAULT_MESH_OPACITY = 0.8
UINT8_MAX = 255.0
MASK_FOREGROUND_VALUE = 255
EXPR_PARAM_DIM = 72


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


def rgba_to_inference_bgr(rgba: np.ndarray) -> np.ndarray:
    """Composite RGBA onto neutral gray and replace magenta chroma-key pixels.

    PIL/numpy RGBA arrays are RGB-ordered. This function composites in RGB
    space, removes magenta, then converts to BGR for OpenCV / HYPIR / SAM3D.
    """
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3] if rgba.shape[2] == 4 else np.full(rgba.shape[:2], 255, dtype=np.uint8)
    neutral_rgb = np.full_like(rgb, NEUTRAL_BG_RGB, dtype=np.uint8)
    alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
    rgb = (rgb.astype(np.float32) * alpha_f + neutral_rgb.astype(np.float32) * (1.0 - alpha_f))
    rgb = rgb.astype(np.uint8)

    magenta_mask = (
        (rgb[:, :, 0] > MAGENTA_THRESHOLD_RGB[0])
        & (rgb[:, :, 1] < MAGENTA_THRESHOLD_RGB[1])
        & (rgb[:, :, 2] > MAGENTA_THRESHOLD_RGB[2])
    )
    rgb[magenta_mask] = NEUTRAL_BG_RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def tight_bbox_xyxy(frame_rgba: np.ndarray) -> np.ndarray:
    """Return a tight xyxy bbox around sprite content, or the full frame if empty."""
    height, width = frame_rgba.shape[:2]
    bbox = tight_content_bbox(frame_rgba, 0, 0, width, height)
    if bbox is None:
        return np.array([[0, 0, width, height]], dtype=np.float32)
    return np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=np.float32)


def scale_bbox_xyxy(bbox: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = bbox.copy()
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    return scaled


def build_inference_mask(frame_rgba: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Full-image uint8 mask for SAM3D mask conditioning (N, H, W, 1)."""
    mask = (content_mask(frame_rgba).astype(np.uint8) * MASK_FOREGROUND_VALUE)
    target_height, target_width = target_shape
    if mask.shape[:2] != (target_height, target_width):
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return mask.reshape(1, target_height, target_width, 1)


def save_display_image(image_bgr: np.ndarray, output_path: Path) -> None:
    """Save a BGR image as an RGB PNG for correct colors in standard viewers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path)


def save_bgr_image(image_bgr: np.ndarray, output_path: Path) -> None:
    save_display_image(image_bgr, output_path)


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
        "pred_global_rots": numpy_to_jsonable(output["pred_global_rots"]),
        "pred_joint_coords": numpy_to_jsonable(output["pred_joint_coords"]),
        "mhr_model_params": numpy_to_jsonable(output.get("mhr_model_params")),
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


def composite_mesh_overlay(
    mesh_rgba: np.ndarray,
    image_bgr: np.ndarray,
    opacity: float,
) -> np.ndarray:
    """Blend a mesh render over the source image at the given opacity."""
    mesh_color = mesh_rgba[:, :, :3].astype(np.float32)
    mesh_alpha = (mesh_rgba[:, :, 3].astype(np.float32) * opacity)[:, :, None]
    background = image_bgr.astype(np.float32) / UINT8_MAX
    blended = mesh_color * mesh_alpha + background * (1.0 - mesh_alpha)
    return np.clip(blended * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)


def frame_name(frame_index: int) -> str:
    return f"frame_{frame_index:0{FRAME_NAME_WIDTH}d}"


def run_inference_on_frame(
    estimator: SAM3DBodyEstimator,
    upscaled_bgr: np.ndarray,
    frame_rgba: np.ndarray,
    use_tight_bbox: bool,
    use_mask: bool,
) -> dict | None:
    height, width = upscaled_bgr.shape[:2]
    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)

    source_height, source_width = frame_rgba.shape[:2]
    scale_x = width / source_width
    scale_y = height / source_height

    if use_tight_bbox:
        bbox = scale_bbox_xyxy(tight_bbox_xyxy(frame_rgba), scale_x, scale_y)
    else:
        bbox = np.array([[0, 0, width, height]], dtype=np.float32)

    masks = build_inference_mask(frame_rgba, (height, width)) if use_mask else None

    outputs = estimator.process_one_image(
        upscaled_rgb,
        bboxes=bbox,
        masks=masks,
        inference_type="body",
    )
    if not outputs:
        return None
    return outputs[0]


def render_frame_outputs(
    output: dict,
    upscaled_bgr: np.ndarray,
    estimator: SAM3DBodyEstimator,
    inference_dir: Path,
    mesh_opacity: float,
) -> None:
    # pred_cam_t/focal_length may have been LERP'd into plain lists/floats by
    # repair_camera_params(), so coerce back to arrays for the renderer.
    renderer = Renderer(focal_length=float(output["focal_length"]), faces=estimator.faces)
    mesh_rgba = render_mesh_transparent(
        renderer,
        output["pred_vertices"],
        np.asarray(output["pred_cam_t"], dtype=np.float32),
        upscaled_bgr,
    )
    save_rgba_image(mesh_rgba, str(inference_dir / "mesh.png"))
    overlay_bgr = composite_mesh_overlay(mesh_rgba, upscaled_bgr, mesh_opacity)
    save_bgr_image(overlay_bgr, inference_dir / "overlay.png")


def lock_sequence_identity(frames: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lock shape/scale identity and freeze hand pose to per-dimension medians."""
    weights = np.ones(len(frames), dtype=np.float32)
    shapes = np.array([frame["shape_params"] for frame in frames], dtype=np.float32)
    scales = np.array([frame["scale_params"] for frame in frames], dtype=np.float32)
    hands = np.array([frame["hand_pose_params"] for frame in frames], dtype=np.float32)

    shape_locked = weighted_median_per_dim(shapes, weights)
    scale_locked = weighted_median_per_dim(scales, weights)
    hand_locked = np.median(hands, axis=0).astype(np.float32)

    for frame in frames:
        frame["shape_params"] = shape_locked.tolist()
        frame["scale_params"] = scale_locked.tolist()
        frame["hand_pose_params"] = hand_locked.tolist()

    return shape_locked, scale_locked, hand_locked


def repair_camera_params(
    frames: list[dict], outlier_indices: set[int], is_loop: bool
) -> None:
    """LERP pred_cam_t/focal_length across repaired outlier runs, mirroring
    repair_outliers' body_pose_params interpolation. Without this, a repaired
    frame keeps its own (possibly bad) original camera while its pose is
    replaced by an interpolated one, so the rendered mesh drifts off the
    sprite silhouette.
    """
    num_frames = len(frames)
    runs = find_outlier_runs(outlier_indices, num_frames, is_loop)
    for run in runs:
        run_start, run_end = run[0], run[-1]
        prev_i = (run_start - 1) % num_frames
        next_i = (run_end + 1) % num_frames
        touches_non_loop_boundary = not is_loop and (run_start == 0 or run_end == num_frames - 1)
        if touches_non_loop_boundary or prev_i in outlier_indices or next_i in outlier_indices:
            continue

        prev_cam_t = np.array(frames[prev_i]["pred_cam_t"], dtype=np.float32)
        next_cam_t = np.array(frames[next_i]["pred_cam_t"], dtype=np.float32)
        prev_focal = float(frames[prev_i]["focal_length"])
        next_focal = float(frames[next_i]["focal_length"])
        num_steps = (next_i - prev_i) % num_frames

        for i in run:
            step = (i - prev_i) % num_frames
            t = step / num_steps
            frames[i]["pred_cam_t"] = ((1.0 - t) * prev_cam_t + t * next_cam_t).tolist()
            frames[i]["focal_length"] = (1.0 - t) * prev_focal + t * next_focal


def stabilize_sequence(
    frames: list[dict],
    head_pose,
    device: str,
    is_loop: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Lock identity, repair outliers, smooth, and re-pose every frame."""
    shape_locked, scale_locked, _ = lock_sequence_identity(frames)

    if len(frames) >= 3:
        global_rots = [
            euler_zyx_to_rotmat(np.array(frame["global_rot"], dtype=np.float32))
            for frame in frames
        ]
        body_poses = [np.array(frame["body_pose_params"], dtype=np.float32) for frame in frames]
        rotation_distances, body_pose_distances = consecutive_distances(
            global_rots, body_poses, is_loop
        )
        outlier_indices = find_outliers(
            rotation_distances, body_pose_distances, len(frames), is_loop
        )
        if outlier_indices:
            print(f"  repairing {len(outlier_indices)} outlier frame(s): {sorted(outlier_indices)}")
            repair_outliers(frames, outlier_indices, is_loop)
            repair_camera_params(frames, outlier_indices, is_loop)
        smooth_character(frames, is_loop)

    zero_expr = np.zeros(EXPR_PARAM_DIM, dtype=np.float32)
    for frame in frames:
        result = repose(
            head_pose,
            global_rot=np.array(frame["global_rot"], dtype=np.float32),
            body_pose_params=np.array(frame["body_pose_params"], dtype=np.float32),
            hand_pose_params=np.array(frame["hand_pose_params"], dtype=np.float32),
            scale_params=np.array(frame["scale_params"], dtype=np.float32),
            shape_params=np.array(frame["shape_params"], dtype=np.float32),
            expr_params=np.array(frame.get("expr_params", zero_expr), dtype=np.float32),
            device=device,
        )
        frame["pred_vertices"] = result["pred_vertices"]
        frame["pred_keypoints_3d"] = result["pred_keypoints_3d"]
        frame["pred_joint_coords"] = result["pred_joint_coords"]
        frame["pred_global_rots"] = result["pred_global_rots"]
        frame["mhr_model_params"] = result["mhr_model_params"]

    return shape_locked, scale_locked


def compute_identity_rest_offsets(
    head_pose,
    shape_params: np.ndarray,
    scale_params: np.ndarray,
    joint_parents: list[int],
    body_pose_dim: int,
    hand_pose_dim: int,
    device: str,
) -> list[np.ndarray]:
    """Parent-relative rest offsets for the locked character identity."""
    zero_rot = np.zeros(3, dtype=np.float32)
    zero_body = np.zeros(body_pose_dim, dtype=np.float32)
    zero_hand = np.zeros(hand_pose_dim, dtype=np.float32)
    zero_expr = np.zeros(EXPR_PARAM_DIM, dtype=np.float32)

    result = repose(
        head_pose,
        global_rot=zero_rot,
        body_pose_params=zero_body,
        hand_pose_params=zero_hand,
        scale_params=scale_params,
        shape_params=shape_params,
        expr_params=zero_expr,
        device=device,
    )
    # pred_joint_coords has CAMERA_AXIS_FLIP applied (see mhr_repose.repose /
    # mhr_head.py's "Camera system difference" comment) but pred_global_rots
    # does not, so undo the flip here to put both in the same native MHR
    # frame before combining them -- otherwise the offsets end up rotated
    # into the wrong frame, producing an upside-down/twisted rest skeleton.
    rest_pos = result["pred_joint_coords"].copy()
    rest_pos[..., CAMERA_AXIS_FLIP] *= -1
    rest_quats = transform.Rotation.from_matrix(result["pred_global_rots"]).as_quat()

    offsets: list[np.ndarray] = []
    for joint_idx, parent_idx in enumerate(joint_parents):
        if parent_idx == -1:
            offsets.append(rest_pos[joint_idx].copy() * METERS_TO_CENTIMETERS)
        else:
            global_offset = (rest_pos[joint_idx] - rest_pos[parent_idx]) * METERS_TO_CENTIMETERS
            parent_quat = rest_quats[parent_idx]
            parent_rotation = transform.Rotation.from_quat(parent_quat)
            offsets.append(parent_rotation.inv().apply(global_offset))
    return offsets


def apply_identity_rest_offsets(
    exporter: BVHExporter,
    head_pose,
    shape_params: np.ndarray,
    scale_params: np.ndarray,
    body_pose_dim: int,
    hand_pose_dim: int,
    device: str,
) -> None:
    offsets = compute_identity_rest_offsets(
        head_pose,
        shape_params,
        scale_params,
        exporter.joint_parents,
        body_pose_dim,
        hand_pose_dim,
        device,
    )
    exporter.rest_offsets = offsets


def process_frame(
    frame_index: int,
    frame_rgba: np.ndarray,
    output_dir: Path,
    estimator: SAM3DBodyEstimator,
    use_upscale: bool,
    hypir_upscaler: HypirUpscaler | None,
    use_tight_bbox: bool,
    use_mask: bool,
    defer_render: bool,
    mesh_opacity: float,
) -> dict | None:
    frame_bgr = rgba_to_inference_bgr(frame_rgba)
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

    output = run_inference_on_frame(
        estimator,
        upscaled_bgr,
        frame_rgba,
        use_tight_bbox=use_tight_bbox,
        use_mask=use_mask,
    )
    if output is None:
        print(f"WARNING: no SAM3D output for {frame_stem}")
        return None

    output["frame_index"] = frame_index
    inference_dir.mkdir(parents=True, exist_ok=True)

    if not defer_render:
        render_frame_outputs(output, upscaled_bgr, estimator, inference_dir, mesh_opacity)

    save_pose_json(output, inference_dir / "pose.json")
    output["_upscaled_bgr"] = upscaled_bgr
    output["_inference_dir"] = inference_dir
    print(f"  {frame_stem} done")
    return output


def rerender_stabilized_frames(
    frames: list[dict],
    estimator: SAM3DBodyEstimator,
    mesh_opacity: float,
) -> None:
    for frame in frames:
        inference_dir = frame["_inference_dir"]
        upscaled_bgr = frame["_upscaled_bgr"]
        render_frame_outputs(frame, upscaled_bgr, estimator, inference_dir, mesh_opacity)
        save_pose_json(frame, inference_dir / "pose.json")


def strip_internal_fields(frames: list[dict]) -> None:
    for frame in frames:
        frame.pop("_upscaled_bgr", None)
        frame.pop("_inference_dir", None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slice one image into frames and run upscale -> SAM3D-Body -> overlay -> BVH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            f"  python scripts/basic_pipeline.py --image input/townsfolk.png "
            f"--rect {DEFAULT_TOWNSFOLK_RECT} --frames 25 --upscale --loop "
            f"--output output/townsfolk_stable"
        ),
    )
    parser.add_argument("--image", required=True, help="Input sprite sheet image path")
    parser.add_argument("--frames", type=int, required=True, help="Number of horizontal frames")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--rect",
        default=None,
        help=f"Optional crop before slicing, as x,y,width,height (townsfolk: {DEFAULT_TOWNSFOLK_RECT})",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale each frame with HYPIR before inference (without this flag, "
        "upscaled/ is a passthrough copy of frames/)",
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=DEFAULT_UPSCALE_FACTOR,
        help="HYPIR upscale factor when --upscale is set",
    )
    parser.add_argument(
        "--mesh-opacity",
        type=float,
        default=DEFAULT_MESH_OPACITY,
        help="Mesh opacity in overlay.png, 0.0 (invisible) to 1.0 (opaque)",
    )
    parser.add_argument(
        "--tight-bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop SAM3D inference to tight sprite content bbox (default: on)",
    )
    parser.add_argument(
        "--use-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass sprite alpha/content mask into SAM3D (default: on)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Treat the frame sequence as cyclic for temporal smoothing",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Skip identity locking and temporal smoothing",
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
    stabilize = not args.no_stabilize

    rgba = load_and_crop_rgba(image_path, rect)
    frame_rgbs = slice_horizontal_frames(rgba, args.frames)
    frame_height, frame_width = frame_rgbs[0].shape[:2]
    print(
        f"Sliced {len(frame_rgbs)} frames of {frame_width}x{frame_height} from {image_path}"
    )

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
            hypir_upscaler=hypir_upscaler,
            use_tight_bbox=args.tight_bbox,
            use_mask=args.use_mask,
            defer_render=stabilize,
            mesh_opacity=args.mesh_opacity,
        )
        if output is not None:
            frame_outputs.append(output)

    if not frame_outputs:
        raise RuntimeError("No frames produced SAM3D output; animation.bvh was not written")

    shape_locked: np.ndarray | None = None
    scale_locked: np.ndarray | None = None
    if stabilize:
        print("Stabilizing sequence (identity lock + temporal smoothing)...")
        shape_locked, scale_locked = stabilize_sequence(
            frame_outputs,
            model.head_pose,
            device=device,
            is_loop=args.loop,
        )
        print("Re-rendering mesh and overlay from stabilized poses...")
        rerender_stabilized_frames(frame_outputs, estimator, args.mesh_opacity)

    bvh_path = output_dir / "animation.bvh"
    exporter = build_exporter(mhr_model=model.head_pose.mhr)
    if shape_locked is not None and scale_locked is not None:
        body_pose_dim = len(frame_outputs[0]["body_pose_params"])
        hand_pose_dim = len(frame_outputs[0]["hand_pose_params"])
        apply_identity_rest_offsets(
            exporter,
            model.head_pose,
            shape_locked,
            scale_locked,
            body_pose_dim,
            hand_pose_dim,
            device,
        )

    strip_internal_fields(frame_outputs)
    write_bvh(frame_outputs, bvh_path, exporter, fps=args.fps)
    print(f"Exported BVH: {bvh_path} ({len(frame_outputs)} frames)")
    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
