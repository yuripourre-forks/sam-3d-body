#!/usr/bin/env python3
"""Re-export animation.bvh files upright, removing the sprite sheet's camera tilt.

SAM3D-Body predicts pose in *camera* space, so a character drawn on an
isometric sheet comes out tilted rather than standing on the floor. This
re-exports each character's BVH with a rigid world-space pre-rotation that
puts its up axis back on +Y, leaving the pose itself (and each character's own
facing direction, which is genuine walking direction, not camera tilt) intact.

Runs off the pose.json files an earlier basic_pipeline.py run already wrote, so
it needs no re-inference. Rendered mesh.png/overlay.png stay in camera space on
purpose -- those have to keep matching the sprites.

A standing character's own body-up direction is the best available estimate of
the tilt, so each one is de-tilted by its own. Sequences the config marks as
non-standing (sitting, lying) have no such reference -- snapping their body-up
to vertical would stand a seated character up -- so they instead get the
sheet-wide tilt estimated from every standing frame.

Usage:
    python scripts/bvh_upright.py --output-root output/townsfolk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.basic_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MHR_PATH,
    apply_identity_rest_offsets,
)
from scripts.export_bvh import DEFAULT_FPS, build_exporter, write_bvh  # noqa: E402
from scripts.mhr_repose import load_head_pose  # noqa: E402
from scripts.mhr_repose import CAMERA_AXIS_FLIP  # noqa: E402
from scripts.pose_priors import robust_shared_up, shortest_arc_rotation  # noqa: E402

DEFAULT_OUTPUT_ROOT = "output/townsfolk"
DEFAULT_CONFIG = "configs/townsfolk.json"
WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
# The character's vertical is measured from its own skeleton rather than from
# pose_priors.BODY_UP_AXIS: that constant is explicitly documented as a rig
# convention *assumption*, which is fine for comparing frames to each other
# (all it needs is to be consistently wrong) but not for aligning to world +Y,
# where being wrong tilts the result. Feet-to-head is measured geometry.
HEAD_JOINT_NAME = "c_head"
FOOT_JOINT_NAMES = ("l_foot", "r_foot")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_character_poses(character_dir: Path) -> list[dict]:
    frames = []
    for pose_path in sorted(character_dir.glob("inference/frame_*/pose.json")):
        frames.append(json.loads(pose_path.read_text(encoding="utf-8")))
    frames.sort(key=lambda frame: frame.get("frame_index", 0))
    return frames


def sequence_up_vectors(frames: list[dict], joint_names: list[str]) -> np.ndarray:
    """Per-frame mid-feet-to-head direction, in the BVH's native rotation frame.

    pose.json's pred_joint_coords carries mhr_head.py's "Camera system
    difference" axis flip while pred_global_rots does not, so the flip is undone
    here to put positions in the same frame the exported rotations live in.
    """
    head_index = joint_names.index(HEAD_JOINT_NAME)
    foot_indices = [joint_names.index(name) for name in FOOT_JOINT_NAMES]

    up_vectors = []
    for frame in frames:
        coords = np.array(frame["pred_joint_coords"], dtype=np.float32)
        coords[..., CAMERA_AXIS_FLIP] *= -1
        up_vectors.append(coords[head_index] - coords[foot_indices].mean(axis=0))
    return np.array(up_vectors, dtype=np.float32)


def tilt_from_vertical_deg(up_vector: np.ndarray) -> float:
    unit = up_vector / np.linalg.norm(up_vector)
    return float(np.degrees(np.arccos(np.clip(np.dot(unit, WORLD_UP), -1.0, 1.0))))


def load_standing_flags(config_path: Path) -> dict[str, bool]:
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        character["name"]: bool(character["sequences"][0].get("standing", True))
        for character in config.get("characters", [])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mhr-path", default=str(DEFAULT_MHR_PATH))
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_root = resolve_path(args.output_root)
    standing_flags = load_standing_flags(resolve_path(args.config))

    print("Loading SAM3D-Body model...")
    head_pose = load_head_pose(
        checkpoint_path=str(resolve_path(args.checkpoint)),
        mhr_path=str(resolve_path(args.mhr_path)),
        device=device,
    )
    joint_names = build_exporter(mhr_model=head_pose.mhr).model_joint_names

    characters = []
    for character_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        frames = load_character_poses(character_dir)
        if not frames:
            continue
        characters.append(
            {
                "dir": character_dir,
                "name": character_dir.name,
                "frames": frames,
                "ups": sequence_up_vectors(frames, joint_names),
                "standing": standing_flags.get(character_dir.name, True),
            }
        )

    if not characters:
        raise SystemExit(f"no pose.json frames found under {output_root}")

    standing_ups = np.concatenate(
        [character["ups"] for character in characters if character["standing"]]
    ) if any(character["standing"] for character in characters) else None
    sheet_up = robust_shared_up(standing_ups) if standing_ups is not None else WORLD_UP
    print(
        f"Sheet camera tilt from {0 if standing_ups is None else len(standing_ups)} "
        f"standing frames: {tilt_from_vertical_deg(sheet_up):.2f} deg"
    )

    print(f"\n{'char':9} {'source':9} {'tilt_before_deg':>15} {'tilt_after_deg':>14}")
    for character in characters:
        frames = character["frames"]
        source_up = (
            robust_shared_up(character["ups"]) if character["standing"] else sheet_up
        )
        world_rotation = shortest_arc_rotation(source_up, WORLD_UP)

        exporter = build_exporter(mhr_model=head_pose.mhr)
        apply_identity_rest_offsets(
            exporter,
            head_pose,
            np.array(frames[0]["shape_params"], dtype=np.float32),
            np.array(frames[0]["scale_params"], dtype=np.float32),
            len(frames[0]["body_pose_params"]),
            len(frames[0]["hand_pose_params"]),
            device,
        )

        bvh_path = character["dir"] / "animation.bvh"
        write_bvh(frames, bvh_path, exporter, fps=args.fps, world_rotation=world_rotation)

        corrected_up = world_rotation @ robust_shared_up(character["ups"])
        print(
            f"{character['name']:9} {'own' if character['standing'] else 'sheet':9} "
            f"{tilt_from_vertical_deg(source_up):15.2f} "
            f"{tilt_from_vertical_deg(corrected_up):14.2f}"
        )

    print(f"\nRe-exported {len(characters)} BVH file(s) upright.")


if __name__ == "__main__":
    main()
