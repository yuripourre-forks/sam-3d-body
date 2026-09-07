#!/usr/bin/env python3
"""Shared utility for re-posing the MHR body model from parameters.

Wraps SAM3DBody's MHRHead.mhr_forward(...) (the exact forward pass used during
original inference) so pipeline stages that edit pose/orientation/shape/scale
parameters (shared-up orientation prior, locked per-character identity,
temporal smoothing) can regenerate self-consistent geometry -- vertices,
keypoints, joint coordinates, per-joint rotations, and the packed MHR model
params -- without re-running the neural network, only the parametric MHR
forward pass it already used to produce the original pose.json outputs.

process_townsfolk.py's default --inference-type is "body" (not "full"), so
every saved pose.json's body_pose_params/hand_pose_params/global_rot/
shape_params/scale_params are exactly the inputs a single MHRHead.forward()
call used to produce pred_vertices/pred_keypoints_3d/etc -- no wrist-IK or
hand-refinement stage (that only runs for --inference-type=full, via the
separate head_pose_hand module). So calling model.head_pose.mhr_forward(...)
with those same saved parameters reproduces the saved geometry almost exactly
(see verify_repose_matches_original below), and editing one or more of those
parameters before the call regenerates correspondingly-updated, self-consistent
geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam_3d_body import load_sam_3d_body  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/model.ckpt"
DEFAULT_MHR_PATH = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"

# MHRHead.forward() flips these two axes on vertices/keypoints/joint_coords right
# after the MHR forward pass ("Camera system difference" in mhr_head.py) -- repose()
# mirrors that exactly so its outputs land in the same space as pose.json's fields.
CAMERA_AXIS_FLIP = [1, 2]
NUM_OUTPUT_KEYPOINTS = 70


def load_head_pose(
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    mhr_path: str = str(DEFAULT_MHR_PATH),
    device: str = "cuda",
):
    """Load the full SAM3D-Body model and return just model.head_pose.

    model.head_pose is the module actually used for --inference-type=body (the
    pipeline's default): it carries all buffers mhr_forward needs (scale_mean/
    scale_comps, hand_pose_mean/comps, keypoint_mapping, the MHR JIT module
    itself), already patched exactly as they were during the original inference
    (e.g. SAM3DBody.__init__ overwrites hand_pose_comps with an identity matrix).
    """
    model, _ = load_sam_3d_body(checkpoint_path=checkpoint_path, mhr_path=mhr_path, device=device)
    model.eval()
    return model.head_pose


def _to_batch(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device).unsqueeze(0)


def repose(
    head_pose,
    global_rot: np.ndarray,
    body_pose_params: np.ndarray,
    hand_pose_params: np.ndarray,
    scale_params: np.ndarray,
    shape_params: np.ndarray,
    expr_params: np.ndarray,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Re-run MHRHead.mhr_forward for one frame's parameters.

    global_trans is always zero (matching MHRHead.forward()'s
    `global_trans = torch.zeros_like(global_rot_euler)` -- root translation is
    handled separately by this pipeline's ground-plane placement, not by MHR).

    Returns un-grounded (ground_offset not applied) geometry in the same
    post-flip space as pose.json's pred_vertices/pred_keypoints_3d/
    pred_joint_coords/pred_global_rots/mhr_model_params.
    """
    global_trans = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        verts, kps308, jcoords, mhr_model_params, joint_global_rots = head_pose.mhr_forward(
            global_trans=global_trans,
            global_rot=_to_batch(global_rot, device),
            body_pose_params=_to_batch(body_pose_params, device),
            hand_pose_params=_to_batch(hand_pose_params, device),
            scale_params=_to_batch(scale_params, device),
            shape_params=_to_batch(shape_params, device),
            expr_params=_to_batch(expr_params, device),
            return_keypoints=True,
            do_pcblend=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )

    verts = verts.clone()
    kps70 = kps308[:, :NUM_OUTPUT_KEYPOINTS].clone()
    jcoords = jcoords.clone()

    verts[..., CAMERA_AXIS_FLIP] *= -1
    kps70[..., CAMERA_AXIS_FLIP] *= -1
    jcoords[..., CAMERA_AXIS_FLIP] *= -1

    return {
        "pred_vertices": verts[0].cpu().numpy(),
        "pred_keypoints_3d": kps70[0].cpu().numpy(),
        "pred_joint_coords": jcoords[0].cpu().numpy(),
        "pred_global_rots": joint_global_rots[0].cpu().numpy(),
        "mhr_model_params": mhr_model_params[0].cpu().numpy(),
    }


def repose_from_pose_json(head_pose, pose_data: dict, device: str = "cuda") -> dict[str, np.ndarray]:
    """Convenience wrapper: repose using a pose.json dict's own (unedited) params.

    Useful both as a smoke test (should reproduce the saved un-grounded geometry)
    and as the common path for stages that only touch a subset of the fields
    (e.g. only global_rot, or only shape_params/scale_params).
    """
    return repose(
        head_pose,
        global_rot=np.array(pose_data["global_rot"], dtype=np.float32),
        body_pose_params=np.array(pose_data["body_pose_params"], dtype=np.float32),
        hand_pose_params=np.array(pose_data["hand_pose_params"], dtype=np.float32),
        scale_params=np.array(pose_data["scale_params"], dtype=np.float32),
        shape_params=np.array(pose_data["shape_params"], dtype=np.float32),
        expr_params=np.array(pose_data["expr_params"], dtype=np.float32),
        device=device,
    )


FOOT_KEYPOINT_INDICES = [15, 16, 17, 18, 19, 20]


def _verify_repose_matches_original(inference_dir: Path, char_id: int, frame_idx: int, device: str) -> None:
    """Regression check that repose() reproduces the original saved geometry.

    Note: pose.json's own "ground_offset" field is *not* the full Y-shift baked
    into its pred_vertices/pred_keypoints_3d/pred_joint_coords. Looking at
    apply_ground_translation, the actual shift applied at save time was
    `compute_ground_offset(this_frame's_own_raw_keypoints) + ground_offset_field`
    -- i.e. this frame's own instantaneous foot-grounding offset *plus* the
    character-level median stored in the field. Only the median half is
    recorded. That's fine for this repo's later stages, which only ever need
    the *parameters* (global_rot/body_pose_params/shape_params/scale_params),
    never pose.json's saved grounded geometry, to regenerate poses -- grounding
    itself is intentionally redone from scratch in Stage 6 (shared ground
    plane). This check reconstructs the missing per-frame component from a
    fresh, unedited repose() call so the comparison is still meaningful.
    """
    import json

    frame_dir = inference_dir / f"char_{char_id:02d}" / f"frame_{frame_idx:03d}"
    with open(frame_dir / "pose.json", encoding="utf-8") as pose_file:
        pose_data = json.load(pose_file)

    head_pose = load_head_pose(device=device)
    fresh_raw = repose_from_pose_json(head_pose, pose_data, device=device)

    fresh_foot_min_y = float(fresh_raw["pred_keypoints_3d"][FOOT_KEYPOINT_INDICES, 1].min())
    full_offset_applied = float(pose_data["ground_offset"]) - fresh_foot_min_y
    offset_vec = np.array([0.0, full_offset_applied, 0.0], dtype=np.float32)

    saved_vertices = np.array(pose_data["pred_vertices"], dtype=np.float32)
    saved_keypoints = np.array(pose_data["pred_keypoints_3d"], dtype=np.float32)

    vert_err = np.abs((fresh_raw["pred_vertices"] + offset_vec) - saved_vertices).max()
    kp_err = np.abs((fresh_raw["pred_keypoints_3d"] + offset_vec) - saved_keypoints).max()
    print(f"reconstructed full ground offset: {full_offset_applied:.6f} (field was {pose_data['ground_offset']:.6f})")
    print(f"max |vertex| repose error: {vert_err:.6f}")
    print(f"max |keypoint_3d| repose error: {kp_err:.6f}")
    assert vert_err < 1e-3, "repose() does not match original pose.json geometry"
    print("OK: repose() reproduces the original saved geometry (mod. the known ground-offset accounting).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test repose() against a saved pose.json")
    parser.add_argument("--inference-dir", default="output/inference")
    parser.add_argument("--char-id", type=int, default=0)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    if not inference_dir.is_absolute():
        inference_dir = REPO_ROOT / inference_dir

    _verify_repose_matches_original(inference_dir, args.char_id, args.frame_idx, args.device)
