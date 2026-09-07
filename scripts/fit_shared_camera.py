#!/usr/bin/env python3
"""Stage 1: fit ONE shared camera + ONE shared metric-to-pixel scale for the sheet.

Currently mesh_render.png is rendered with a hardcoded frontal camera
(elevation=0, azimuth=0) and each frame is independently rescaled to fill its
own canvas (render_mesh_isometric normalizes by that frame's own
`max ||v - mean||`). Neither uses the fact that the sprite sheet was drawn with
one fixed isometric camera: this script fits (elevation_deg, azimuth_deg,
roll_deg, scale_factor) once, for the whole 160-frame sheet, by maximizing mean
silhouette IoU against the sprites' own masks, and persists the result to
output/camera.json for later stages (ground-and-nudge, verify) to render with
instead of the hardcoded frontal camera.

Camera *direction* (elevation_deg/azimuth_deg/roll_deg) is fit once, shared by
every character -- the sprite sheet really was drawn from one fixed isometric
viewpoint, so this part of "one shared camera" is physically meaningful.

Size is a different matter: pixel art routinely draws characters at different
sizes on purpose (bulkier armor, a seated character occupying a shorter
panel, a hunched/child NPC) independent of camera geometry, so forcing one
scale_factor on every character just trades one character's fit for
another's without actually being correct for either -- confirmed empirically
here (a single global scale_factor made the previously-good characters worse
while barely helping the previously-bad ones; see git history / camera_v*
snapshots in output/). So scale is fit per character (fit_per_character_scale_
factors, after the shared direction is fixed): effective_extent[char] =
extent_reference[char] / per_character_scale_factor[char]. extent_reference
itself is still pose-invariant per character (see below) so a character's
per-character scale_factor purely captures "the artist drew this character at
this size", not any leftover pose-dependent artifact.

`scale_factor` (singular, camera-direction-fit-time) is kept in the output for
reference/debugging only; scripts/render_final_poses.py uses
`per_character_scale_factor` for actual rendering.

extent_reference itself must be pose-invariant (a property of *who* the
character is, not what they're doing in any given frame), otherwise a
character whose sprite is seated or lying down for its whole animation gets a
systematically wrong on-screen size relative to standing characters. It is
therefore computed from a canonical rest pose (body_pose_params and
hand_pose_params zeroed, global_rot zeroed) built from that character's own
locked shape_params/scale_params (see scripts/repose_with_priors.py) via
scripts/mhr_repose.py, rather than from any frame's actual (pose-dependent)
`max ||v - centroid||`. A seated or prone character now renders smaller than a
standing one only when its *current frame's pose* is more compact -- not
because its rest-pose body size was mismeasured from a curled-up frame.

Search strategy (coarse-to-fine, as suggested by the plan): a coarse grid
search over all four parameters seeded at the assumed isometric values
(elevation=30, azimuth=45), followed by coordinate-descent refinement (a finer
1D line search per axis, repeated for a couple of rounds) around the coarse
optimum. A handful of frames per character (not all 160) are used for fitting
to keep the search tractable; the camera itself should be pose-invariant, so a
small representative sample per character is enough.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.foot_grounding import (  # noqa: E402
    content_bbox_and_centroid_from_mask,
    render_mesh_centroid_placed,
    render_mesh_frontal_placed,
)
from scripts.manifest_utils import inference_frame_dir_index  # noqa: E402
from scripts.mhr_repose import load_head_pose, repose  # noqa: E402
from scripts.pose_metrics import composite_score, mask_iou, sprite_mask_at_size  # noqa: E402

SEED_ELEVATION_DEG = 30.0
SEED_AZIMUTH_DEG = 45.0
SEED_ROLL_DEG = 0.0
SEED_SCALE_FACTOR = 1.0

COARSE_ELEVATION_RANGE = (10.0, 50.0, 5)  # (low, high, num_steps)
COARSE_AZIMUTH_RANGE = (15.0, 75.0, 5)

REFINE_ROUNDS = 2
REFINE_HALF_WIDTHS = {"elevation_deg": 6.0, "azimuth_deg": 6.0, "roll_deg": 4.0, "scale_factor": 0.12}
REFINE_STEPS = 5

FRAMES_PER_CHARACTER = 2
FIT_CANVAS_SCALE = 2  # small render scale during search; final rendering uses its own render_scale


def mesh_extent(vertices: np.ndarray) -> float:
    """max ||v - centroid|| -- the same normalizer render_mesh_isometric uses by default."""
    centered = vertices - vertices.mean(axis=0)
    return float(np.max(np.linalg.norm(centered, axis=1)))


def load_manifest(split_dir: Path) -> dict:
    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def load_faces(inference_dir: Path, manifest: dict) -> np.ndarray:
    for char_entry in manifest["characters"]:
        for frame_info in char_entry["frames"]:
            mesh_path = inference_dir / f"char_{char_entry['id']:02d}" / f"frame_{inference_frame_dir_index(frame_info):03d}" / "mesh.ply"
            if mesh_path.exists():
                return trimesh.load(str(mesh_path), process=False).faces
    raise FileNotFoundError("No mesh.ply found under inference_dir to read face topology from")


BOUNDARY_LAMBDA = 0.002


def load_non_standing_sequence_keys(reposed_dir: Path) -> set[tuple[int, str]]:
    report_path = reposed_dir / "locked_identity_report.json"
    if not report_path.exists():
        return set()
    with open(report_path, encoding="utf-8") as report_file:
        report = json.load(report_file)
    keys: set[tuple[int, str]] = set()
    for char_id, info in report.get("characters", {}).items():
        sequences = info.get("sequences")
        if sequences:
            for sequence_name, sequence_info in sequences.items():
                if not sequence_info.get("upright_constraint_applied", True):
                    keys.add((int(char_id), sequence_name))
        elif not info.get("upright_constraint_applied", True):
            keys.add((int(char_id), "default"))
    return keys


def load_non_standing_char_ids(reposed_dir: Path) -> set[int]:
    return {char_id for char_id, _ in load_non_standing_sequence_keys(reposed_dir)}


def load_pose(inference_dir: Path, char_id: int, frame_idx: int) -> dict | None:
    pose_path = inference_dir / f"char_{char_id:02d}" / f"frame_{frame_idx:03d}" / "pose.json"
    if not pose_path.exists():
        return None
    with open(pose_path, encoding="utf-8") as pose_file:
        return json.load(pose_file)


def find_first_pose(inference_dir: Path, char_entry: dict) -> dict | None:
    for frame_info in char_entry["frames"]:
        pose_data = load_pose(inference_dir, char_entry["id"], inference_frame_dir_index(frame_info))
        if pose_data is not None:
            return pose_data
    return None


def compute_rest_pose_extent_references(
    inference_dir: Path, manifest: dict, device: str
) -> dict[int, float]:
    """One pose-invariant extent per character (see module docstring).

    Only needs a single frame per character: once identity is locked (Stage 3),
    shape_params/scale_params are identical across that character's whole
    animation, so any one frame's saved values fully determine its rest pose.
    """
    head_pose = load_head_pose(device=device)
    extent_reference = {}
    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        pose_data = find_first_pose(inference_dir, char_entry)
        if pose_data is None:
            extent_reference[char_id] = 1.0
            continue

        shape_params = np.array(pose_data["shape_params"], dtype=np.float32)
        scale_params = np.array(pose_data["scale_params"], dtype=np.float32)
        expr_params = np.array(pose_data["expr_params"], dtype=np.float32)
        body_pose_params = np.zeros_like(np.array(pose_data["body_pose_params"], dtype=np.float32))
        hand_pose_params = np.zeros_like(np.array(pose_data["hand_pose_params"], dtype=np.float32))
        global_rot = np.zeros(3, dtype=np.float32)

        rest = repose(
            head_pose,
            global_rot=global_rot,
            body_pose_params=body_pose_params,
            hand_pose_params=hand_pose_params,
            scale_params=scale_params,
            shape_params=shape_params,
            expr_params=expr_params,
            device=device,
        )
        extent_reference[char_id] = mesh_extent(rest["pred_vertices"])
    return extent_reference


class FitSample:
    __slots__ = (
        "char_id",
        "frame_idx",
        "vertices",
        "frame_path",
        "canvas_width",
        "canvas_height",
        "sequence_name",
        "view_yaw_deg",
        "angle_index",
    )

    def __init__(
        self,
        char_id,
        frame_idx,
        vertices,
        frame_path,
        canvas_width,
        canvas_height,
        sequence_name="default",
        view_yaw_deg=0.0,
        angle_index=0,
    ):
        self.char_id = char_id
        self.frame_idx = frame_idx
        self.vertices = vertices
        self.frame_path = frame_path
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.sequence_name = sequence_name
        self.view_yaw_deg = view_yaw_deg
        self.angle_index = angle_index


def select_fit_samples(
    split_dir: Path, inference_dir: Path, manifest: dict, frames_per_character: int
) -> list[FitSample]:
    """Pick up to `frames_per_character` representative samples per
    (character, sequence, angle) group, not per raw character-frame-list
    position. For single-angle sheets (townsfolk) this reduces to one group
    per character -- identical to the old behavior. For multi-angle sheets
    (succubus) this is required: a single character can span hundreds of
    frames across many sequences/angles, so picking `frames_per_character`
    samples from the *whole character* would starve most (sequence, angle)
    groups of any data at all, both for the shared direction fit and for
    fit_per_angle_nudge_deg's per-group nudge search.
    """
    from collections import defaultdict

    samples = []
    for char_entry in manifest["characters"]:
        char_id = char_entry["id"]
        frame_infos = char_entry["frames"]
        if not frame_infos:
            continue

        groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for frame_info in frame_infos:
            key = (
                frame_info.get("sequence_name", f"char_{char_id:02d}"),
                int(frame_info.get("angle_index", 0)),
            )
            groups[key].append(frame_info)

        for (sequence_name, angle_index), group_frame_infos in sorted(groups.items()):
            group_frame_infos = sorted(
                group_frame_infos,
                key=lambda info: int(info.get("timestep_index", info["index"])),
            )
            num_picks = min(frames_per_character, len(group_frame_infos))
            pick_indices = sorted(
                {int(round(i)) for i in np.linspace(0, len(group_frame_infos) - 1, num_picks)}
            )
            for pick in pick_indices:
                frame_info = group_frame_infos[pick]
                pose_data = load_pose(inference_dir, char_id, inference_frame_dir_index(frame_info))
                if pose_data is None:
                    continue
                samples.append(
                    FitSample(
                        char_id=char_id,
                        frame_idx=frame_info["index"],
                        vertices=np.array(pose_data["pred_vertices"], dtype=np.float32),
                        frame_path=split_dir / frame_info["path"],
                        canvas_width=manifest.get("output_frame_size", manifest.get("frame_size", 96))
                        * FIT_CANVAS_SCALE,
                        canvas_height=manifest.get("output_frame_size", manifest.get("frame_size", 96))
                        * FIT_CANVAS_SCALE,
                        sequence_name=sequence_name,
                        view_yaw_deg=float(frame_info.get("view_yaw_deg", 0.0)),
                        angle_index=angle_index,
                    )
                )
    return samples


def evaluate_camera(
    params: dict,
    samples: list[FitSample],
    extent_reference: dict[int, float],
    faces: np.ndarray,
    sprite_mask_cache: dict[str, np.ndarray],
    per_character_scale_factor: dict[int, float] | None = None,
    angle_nudge_deg: float = 0.0,
) -> float:
    ious = []
    for sample in samples:
        scale_factor = (
            per_character_scale_factor[sample.char_id]
            if per_character_scale_factor is not None
            else params["scale_factor"]
        )
        extent_override = extent_reference[sample.char_id] / scale_factor
        # render_mesh_frontal_placed requires a target; build one from the sprite's own
        # content bbox/centroid at this sample's fit-canvas resolution (cached per frame).
        cache_key = str(sample.frame_path)
        sprite_mask = sprite_mask_cache.get(cache_key)
        if sprite_mask is None:
            sprite_mask = sprite_mask_at_size(sample.frame_path, (sample.canvas_width, sample.canvas_height))
            sprite_mask_cache[cache_key] = sprite_mask

        target = content_bbox_and_centroid_from_mask(sprite_mask)
        if target is None:
            continue

        mesh_rgba = render_mesh_centroid_placed(
            sample.vertices,
            faces,
            canvas_width=sample.canvas_width,
            canvas_height=sample.canvas_height,
            elevation_deg=params["elevation_deg"],
            azimuth_deg=params["azimuth_deg"],
            roll_deg=params["roll_deg"],
            extent_override=extent_override,
            target_centroid_x=target.centroid[0],
            target_centroid_y=target.centroid[1],
            view_yaw_deg=sample.view_yaw_deg,
            angle_nudge_deg=angle_nudge_deg,
        )
        mesh_mask = mesh_rgba[:, :, 3] > 0
        score = composite_score(sprite_mask, mesh_mask, boundary_lambda=BOUNDARY_LAMBDA)
        if score is not None:
            ious.append(score)
    return float(np.mean(ious)) if ious else 0.0


def grid_values(range_spec: tuple[float, float, int]) -> list[float]:
    low, high, steps = range_spec
    return list(np.linspace(low, high, steps))


def coarse_search(samples, extent_reference, faces, sprite_mask_cache) -> tuple[dict, float]:
    """Coarse grid over elevation/azimuth only (the two dominant shape parameters),
    holding roll/scale at their seed values -- a full 4D coarse grid is too expensive
    to render (hundreds of full-sample evaluations); roll and scale are refined
    properly afterward by coordinate_descent_refine over all four axes.
    """
    best_params, best_iou = None, -1.0
    combos = list(itertools.product(grid_values(COARSE_ELEVATION_RANGE), grid_values(COARSE_AZIMUTH_RANGE)))
    total = len(combos)
    for i, (elevation, azimuth) in enumerate(combos):
        params = {
            "elevation_deg": elevation,
            "azimuth_deg": azimuth,
            "roll_deg": SEED_ROLL_DEG,
            "scale_factor": SEED_SCALE_FACTOR,
        }
        iou = evaluate_camera(params, samples, extent_reference, faces, sprite_mask_cache)
        if iou > best_iou:
            best_iou, best_params = iou, params
        print(f"  coarse search {i + 1}/{total}: iou={iou:.4f} (best_iou={best_iou:.4f} best={best_params})")
    return best_params, best_iou


def coordinate_descent_refine(
    params: dict, best_iou: float, samples, extent_reference, faces, sprite_mask_cache
) -> tuple[dict, float]:
    current_params = dict(params)
    current_iou = best_iou
    for round_idx in range(REFINE_ROUNDS):
        for axis, half_width in REFINE_HALF_WIDTHS.items():
            center = current_params[axis]
            candidates = np.linspace(center - half_width, center + half_width, REFINE_STEPS)
            if axis == "scale_factor":
                candidates = candidates[candidates > 0.05]
            for value in candidates:
                trial_params = dict(current_params)
                trial_params[axis] = float(value)
                iou = evaluate_camera(trial_params, samples, extent_reference, faces, sprite_mask_cache)
                if iou > current_iou:
                    current_iou, current_params = iou, trial_params
            print(f"  refine round {round_idx + 1}, axis={axis}: best_iou={current_iou:.4f} best={current_params}")
        # Halve the search half-widths each round for finer refinement.
        for axis in REFINE_HALF_WIDTHS:
            REFINE_HALF_WIDTHS[axis] *= 0.5
    return current_params, current_iou


PER_CHARACTER_SCALE_RANGE = (0.3, 3.5, 33)
ANGLE_NUDGE_OPTIONS_DEG = [-3.0, -1.5, 0.0, 1.5, 3.0]


def fit_per_angle_nudge_deg(
    shared_params: dict,
    samples: list[FitSample],
    extent_reference: dict[int, float],
    per_character_scale_factor: dict[int | str, float],
    faces: np.ndarray,
    sprite_mask_cache: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    """Fit one bounded azimuth nudge per (character, sequence, angle)."""
    from collections import defaultdict

    scale_lookup = {int(key): float(value) for key, value in per_character_scale_factor.items()}
    groups: dict[tuple[int, str, int], list[FitSample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.char_id, sample.sequence_name, sample.angle_index)].append(sample)

    per_angle_nudge: dict[str, dict[str, dict[str, float]]] = {}
    for (char_id, sequence_name, angle_index), group_samples in sorted(groups.items()):
        best_nudge = 0.0
        best_score = -1.0
        for nudge_deg in ANGLE_NUDGE_OPTIONS_DEG:
            score = evaluate_camera(
                shared_params,
                group_samples,
                extent_reference,
                faces,
                sprite_mask_cache,
                per_character_scale_factor=scale_lookup,
                angle_nudge_deg=nudge_deg,
            )
            if score > best_score:
                best_score = score
                best_nudge = nudge_deg
        per_angle_nudge.setdefault(str(char_id), {}).setdefault(sequence_name, {})[
            str(angle_index)
        ] = float(best_nudge)
    return per_angle_nudge


def evaluate_character_scale(
    scale_factor: float,
    char_samples: list[FitSample],
    extent_reference: float,
    shared_params: dict,
    faces: np.ndarray,
    sprite_mask_cache: dict[str, np.ndarray],
) -> float:
    """Mean IoU for one character at one candidate scale_factor.

    Deliberately uses render_mesh_centroid_placed (translate-only placement),
    *not* render_mesh_frontal_placed -- the latter always rescales its render
    to exactly cover the target bbox, which would silently normalize away
    whatever extent_override/scale_factor this is trying to measure (verified
    empirically: fitting scale_factor through render_mesh_frontal_placed
    produced per-character values that catastrophically degraded IoU once
    actually applied at render time via the non-rescaling
    render_mesh_shared_placed -- rescale-based fitting was measuring
    clipping/resampling noise, not real on-screen size preference).
    """
    extent_override = extent_reference / scale_factor
    ious = []
    for sample in char_samples:
        cache_key = str(sample.frame_path)
        sprite_mask = sprite_mask_cache.get(cache_key)
        if sprite_mask is None:
            sprite_mask = sprite_mask_at_size(sample.frame_path, (sample.canvas_width, sample.canvas_height))
            sprite_mask_cache[cache_key] = sprite_mask

        target = content_bbox_and_centroid_from_mask(sprite_mask)
        if target is None:
            continue

        mesh_rgba = render_mesh_centroid_placed(
            sample.vertices,
            faces,
            canvas_width=sample.canvas_width,
            canvas_height=sample.canvas_height,
            elevation_deg=shared_params["elevation_deg"],
            azimuth_deg=shared_params["azimuth_deg"],
            roll_deg=shared_params["roll_deg"],
            extent_override=extent_override,
            target_centroid_x=target.centroid[0],
            target_centroid_y=target.centroid[1],
            view_yaw_deg=sample.view_yaw_deg,
        )
        mesh_mask = mesh_rgba[:, :, 3] > 0
        score = composite_score(sprite_mask, mesh_mask, boundary_lambda=BOUNDARY_LAMBDA)
        if score is not None:
            ious.append(score)
    return float(np.mean(ious)) if ious else 0.0


def fit_per_character_scale_factors(
    shared_params: dict,
    samples: list[FitSample],
    extent_reference: dict[int, float],
    faces: np.ndarray,
    sprite_mask_cache: dict[str, np.ndarray],
) -> dict[int, float]:
    """Per-character size, camera direction still shared.

    A single elevation/azimuth/roll is physically correct (the sprite sheet was
    drawn from one fixed isometric camera), but forcing one metric-to-pixel
    scale on top of that assumes every character is drawn at the same true
    height -- pixel art frequently is not (bulkier armor, a child/hunched NPC,
    a seated character occupying a shorter panel, etc. are all legitimately
    drawn at different sizes by the artist, independent of camera geometry).
    So each character gets its own scale_factor here, searched to maximize
    that character's own mean IoU, while elevation/azimuth/roll stay fixed at
    the values fit above -- this is what actually addresses "characters have
    different heights" instead of forcing a single compromise size on all of
    them.
    """
    scale_candidates = grid_values(PER_CHARACTER_SCALE_RANGE)
    per_character_scale: dict[int, float] = {}
    for char_id, reference in extent_reference.items():
        char_samples = [sample for sample in samples if sample.char_id == char_id]
        if not char_samples:
            per_character_scale[char_id] = shared_params["scale_factor"]
            continue
        best_scale, best_iou = shared_params["scale_factor"], -1.0
        for scale_factor in scale_candidates:
            iou = evaluate_character_scale(
                float(scale_factor), char_samples, reference, shared_params, faces, sprite_mask_cache
            )
            if iou > best_iou:
                best_iou, best_scale = iou, float(scale_factor)

        # Finer 1D refinement around the coarse optimum (half the coarse grid step on
        # either side), same coordinate-descent spirit as the shared-camera fit above.
        coarse_step = (PER_CHARACTER_SCALE_RANGE[1] - PER_CHARACTER_SCALE_RANGE[0]) / (
            PER_CHARACTER_SCALE_RANGE[2] - 1
        )
        for value in np.linspace(best_scale - coarse_step, best_scale + coarse_step, 9):
            if value <= 0:
                continue
            iou = evaluate_character_scale(
                float(value), char_samples, reference, shared_params, faces, sprite_mask_cache
            )
            if iou > best_iou:
                best_iou, best_scale = iou, float(value)

        at_boundary = best_scale <= PER_CHARACTER_SCALE_RANGE[0] + 1e-6 or best_scale >= PER_CHARACTER_SCALE_RANGE[1] - 1e-6
        boundary_note = " (AT SEARCH BOUNDARY -- widen PER_CHARACTER_SCALE_RANGE)" if at_boundary else ""
        per_character_scale[char_id] = best_scale
        print(f"  char_{char_id:02d}: best per-character scale_factor={best_scale:.3f} (iou={best_iou:.4f}){boundary_note}")
    return per_character_scale


def report_local_landscape(
    params: dict, samples, extent_reference, faces, sprite_mask_cache
) -> dict:
    """IoU of the neighbors around the chosen optimum in each dimension (risk mitigation)."""
    landscape = {}
    steps = {"elevation_deg": 5.0, "azimuth_deg": 5.0, "roll_deg": 3.0, "scale_factor": 0.05}
    for axis, step in steps.items():
        neighbor_ious = {}
        for delta in (-step, 0.0, step):
            trial_params = dict(params)
            trial_params[axis] = params[axis] + delta
            neighbor_ious[f"{delta:+.2f}"] = evaluate_camera(
                trial_params, samples, extent_reference, faces, sprite_mask_cache
            )
        landscape[axis] = neighbor_ious
    return landscape


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one shared isometric camera for the whole sprite sheet")
    parser.add_argument(
        "--split-dir",
        default="output/split",
    )
    parser.add_argument(
        "--inference-dir",
        default="output/inference_smoothed",
        help="Source of per-frame pose params (identity-locked/smoothed data gives the most accurate fit)",
    )
    parser.add_argument(
        "--faces-source-dir",
        default="output/inference",
        help="Any inference dir containing mesh.ply, just to read MHR face topology",
    )
    parser.add_argument(
        "--reposed-dir",
        default="output/inference_reposed",
        help="Source of locked_identity_report.json, to find non-standing characters to "
        "exclude from the shared elevation/azimuth/roll direction fit (see "
        "load_non_standing_char_ids)",
    )
    parser.add_argument("--output", default="output/camera.json")
    parser.add_argument("--frames-per-character", type=int, default=FRAMES_PER_CHARACTER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fixed-camera-json",
        default=None,
        help=(
            "Skip the (slow) shared elevation/azimuth/roll/scale_factor coarse+refine "
            "search and reuse those four values from this existing camera.json -- only "
            "extent_reference and per_character_scale_factor are recomputed. Useful for "
            "quickly re-fitting per-character scale after widening its search range."
        ),
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir
    inference_dir = Path(args.inference_dir)
    if not inference_dir.is_absolute():
        inference_dir = REPO_ROOT / inference_dir
    faces_source_dir = Path(args.faces_source_dir)
    if not faces_source_dir.is_absolute():
        faces_source_dir = REPO_ROOT / faces_source_dir
    reposed_dir = Path(args.reposed_dir)
    if not reposed_dir.is_absolute():
        reposed_dir = REPO_ROOT / reposed_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    manifest = load_manifest(split_dir)
    faces = load_faces(faces_source_dir, manifest)
    extent_reference = compute_rest_pose_extent_references(inference_dir, manifest, args.device)
    print(f"Rest-pose extent_reference per character: {extent_reference}")
    samples = select_fit_samples(split_dir, inference_dir, manifest, args.frames_per_character)
    print(f"Fitting camera on {len(samples)} sample frames across {len(manifest['characters'])} characters")

    non_standing_sequence_keys = load_non_standing_sequence_keys(reposed_dir)
    non_standing_char_ids = {char_id for char_id, _ in non_standing_sequence_keys}
    direction_samples = [
        sample
        for sample in samples
        if (sample.char_id, sample.sequence_name) not in non_standing_sequence_keys
    ]
    if non_standing_sequence_keys:
        print(
            f"Excluding non-standing sequence(s) {sorted(non_standing_sequence_keys)} from the shared "
            f"elevation/azimuth/roll direction fit ({len(samples) - len(direction_samples)} of "
            f"{len(samples)} samples excluded); they still get their own per-character scale fit below."
        )

    sprite_mask_cache: dict[str, np.ndarray] = {}

    seed_params = {
        "elevation_deg": SEED_ELEVATION_DEG,
        "azimuth_deg": SEED_AZIMUTH_DEG,
        "roll_deg": SEED_ROLL_DEG,
        "scale_factor": SEED_SCALE_FACTOR,
    }
    if args.fixed_camera_json is not None:
        fixed_path = Path(args.fixed_camera_json)
        if not fixed_path.is_absolute():
            fixed_path = REPO_ROOT / fixed_path
        with open(fixed_path, encoding="utf-8") as fixed_file:
            fixed_camera = json.load(fixed_file)
        final_params = {
            "elevation_deg": fixed_camera["elevation_deg"],
            "azimuth_deg": fixed_camera["azimuth_deg"],
            "roll_deg": fixed_camera["roll_deg"],
            "scale_factor": fixed_camera["scale_factor"],
        }
        seed_iou = evaluate_camera(seed_params, direction_samples, extent_reference, faces, sprite_mask_cache)
        final_iou = evaluate_camera(final_params, direction_samples, extent_reference, faces, sprite_mask_cache)
        print(f"Reusing fixed shared camera from {fixed_path}: {final_params} iou={final_iou:.4f}")
        landscape = fixed_camera.get("local_landscape", {})
    else:
        seed_iou = evaluate_camera(seed_params, direction_samples, extent_reference, faces, sprite_mask_cache)
        print(f"Seed camera {seed_params}: iou={seed_iou:.4f}")

        print("Running coarse grid search...")
        coarse_params, coarse_iou = coarse_search(direction_samples, extent_reference, faces, sprite_mask_cache)
        if seed_iou > coarse_iou:
            coarse_params, coarse_iou = seed_params, seed_iou
        print(f"Coarse best: {coarse_params} iou={coarse_iou:.4f}")

        print("Running coordinate-descent refinement...")
        final_params, final_iou = coordinate_descent_refine(
            coarse_params, coarse_iou, direction_samples, extent_reference, faces, sprite_mask_cache
        )
        print(f"Final: {final_params} iou={final_iou:.4f}")

        landscape = report_local_landscape(final_params, direction_samples, extent_reference, faces, sprite_mask_cache)
        print("Local IoU landscape around the optimum (risk check for a shallow/wrong optimum):")
        for axis, neighbors in landscape.items():
            print(f"  {axis}: {neighbors}")

    print("Fitting per-character scale (camera direction stays shared; size does not)...")
    per_character_scale_factor = fit_per_character_scale_factors(
        final_params, samples, extent_reference, faces, sprite_mask_cache
    )
    per_character_iou = {
        char_id: evaluate_character_scale(
            scale_factor,
            [sample for sample in samples if sample.char_id == char_id],
            extent_reference[char_id],
            final_params,
            faces,
            sprite_mask_cache,
        )
        for char_id, scale_factor in per_character_scale_factor.items()
    }
    per_character_mean_iou = float(np.mean(list(per_character_iou.values()))) if per_character_iou else None
    print(f"Mean IoU using per-character scale: {per_character_mean_iou:.4f} (vs shared scale_factor: {final_iou:.4f})")

    print("Fitting per-(character, sequence, angle) azimuth nudge...")
    per_angle_nudge_deg = fit_per_angle_nudge_deg(
        final_params,
        samples,
        extent_reference,
        per_character_scale_factor,
        faces,
        sprite_mask_cache,
    )

    camera_data = {
        **final_params,
        "fit_mean_iou": final_iou,
        "seed_mean_iou": seed_iou,
        "extent_reference": extent_reference,
        "per_character_scale_factor": per_character_scale_factor,
        "per_character_fit_iou": per_character_iou,
        "per_character_fit_mean_iou": per_character_mean_iou,
        "per_angle_nudge_deg": per_angle_nudge_deg,
        "num_fit_samples": len(samples),
        "num_direction_fit_samples": len(direction_samples),
        "non_standing_sequence_keys_excluded_from_direction_fit": [
            [char_id, sequence_name] for char_id, sequence_name in sorted(non_standing_sequence_keys)
        ],
        "local_landscape": landscape,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as camera_file:
        json.dump(camera_data, camera_file, indent=2)
    print(f"Wrote shared camera: {output_path}")


if __name__ == "__main__":
    main()
