#!/usr/bin/env python3
"""Rotation-matrix-space helpers for the shared-up orientation prior (Stage 2)
and the IoU-weighted locked identity (Stage 3) of the fixed-camera-and-
sequence-priors plan.

Everything here works directly with 3x3 rotation matrices rather than Euler
angles so no assumption about the meaning of individual Euler axes is needed --
only the roundtrip convention (roma's "ZYX", the same one mhr_head.py uses to
decode the network's predicted global rotation) has to be consistent, and that
is verified by scripts/mhr_repose.py's repose() regression check.
"""

from __future__ import annotations

import numpy as np
import roma
import torch

# The character's own "up" axis in its rest pose, before global_rot is applied.
# This is a body-rig convention assumption (most such rigs are Y-up in rest
# pose), not something read out of MHR metadata; robust_shared_up() below only
# ever *compares* these vectors to each other, so as long as the assumption is
# consistently wrong-or-right across every frame the relative correction is
# unaffected -- what matters is that every frame agrees on which body axis
# "up" is, which is trivially true here since it's the same constant for all.
BODY_UP_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float32)

TRIM_ANGLE_DEG = 40.0
TRIM_ITERATIONS = 3
MIN_KEPT_FRACTION = 0.25


def euler_zyx_to_rotmat(global_rot: np.ndarray) -> np.ndarray:
    """(3,) ZYX Euler radians -> (3, 3) rotation matrix, matching mhr_head.py's convention."""
    tensor = torch.as_tensor(np.asarray(global_rot, dtype=np.float32)).unsqueeze(0)
    return roma.euler_to_rotmat("ZYX", tensor)[0].numpy()


def rotmat_to_euler_zyx(rotmat: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix -> (3,) ZYX Euler radians, inverse of euler_zyx_to_rotmat."""
    tensor = torch.as_tensor(np.asarray(rotmat, dtype=np.float32)).unsqueeze(0)
    return roma.rotmat_to_euler("ZYX", tensor)[0].numpy()


def local_up_in_world(rotmat: np.ndarray) -> np.ndarray:
    """Where the character's own rest-pose up axis ends up after `rotmat`."""
    return rotmat @ BODY_UP_AXIS


def robust_shared_up(up_vectors: np.ndarray) -> np.ndarray:
    """Trimmed spherical mean of a (N, 3) batch of unit "up" vectors.

    Iteratively re-estimates the mean direction, discarding vectors more than
    TRIM_ANGLE_DEG away from the current estimate, so a character with
    systematically noisy orientation (e.g. char_06's up-to-38-degree global_rot
    std, per the investigation this came out of) cannot drag the shared
    estimate away from the consensus of every other frame.
    """
    normalized = up_vectors / np.linalg.norm(up_vectors, axis=1, keepdims=True)
    current = normalized.mean(axis=0)
    current /= np.linalg.norm(current)

    min_kept = max(3, int(round(len(normalized) * MIN_KEPT_FRACTION)))
    for _ in range(TRIM_ITERATIONS):
        cos_angles = np.clip(normalized @ current, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angles))
        keep = angles_deg <= TRIM_ANGLE_DEG
        if keep.sum() < min_kept:
            order = np.argsort(angles_deg)
            keep = np.zeros(len(normalized), dtype=bool)
            keep[order[:min_kept]] = True

        new_mean = normalized[keep].mean(axis=0)
        norm = np.linalg.norm(new_mean)
        if norm < 1e-8:
            break
        current = new_mean / norm
    return current


def shortest_arc_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix R such that R @ source == target (both unit vectors)."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cos_angle = float(np.clip(np.dot(source, target), -1.0, 1.0))
    axis = np.cross(source, target)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < 1e-8:
        if cos_angle > 0:
            return np.eye(3, dtype=np.float32)
        # Antiparallel: rotate 180 degrees about any axis perpendicular to source.
        perpendicular = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(source[0]) > 0.9:
            perpendicular = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(source, perpendicular)
        axis /= np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = axis / axis_norm
        angle = float(np.arccos(cos_angle))

    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float32,
    )
    return (
        np.eye(3, dtype=np.float32) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    ).astype(np.float32)


def apply_upright_constraint(global_rot: np.ndarray, shared_up: np.ndarray) -> np.ndarray:
    """Force this frame's local-up-in-world onto shared_up, keeping yaw/twist otherwise.

    Snapping the "up" vector via the minimal (shortest-arc) rotation removes
    exactly the tilt component and nothing else -- no assumption about which
    Euler axis is "yaw" vs "tilt" is needed, sidestepping any ambiguity about
    the MHR/roma Euler convention's axis semantics.
    """
    rotmat = euler_zyx_to_rotmat(global_rot)
    current_up = local_up_in_world(rotmat)
    correction = shortest_arc_rotation(current_up, shared_up)
    upright_rotmat = correction @ rotmat
    return rotmat_to_euler_zyx(upright_rotmat)


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal (e1, e2) basis spanning the plane perpendicular to `axis`."""
    axis = axis / np.linalg.norm(axis)
    arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, arbitrary))) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e1 = np.cross(axis, arbitrary)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1.astype(np.float32), e2.astype(np.float32)


def extract_yaw_angle(rotmat: np.ndarray, shared_up: np.ndarray, reference_rotmat: np.ndarray) -> float:
    """Signed angle (radians) of the rotation-about-shared_up that takes
    reference_rotmat to `rotmat`.

    Valid whenever both rotmat and reference_rotmat already satisfy
    local_up_in_world(R) == shared_up (true for every frame after
    apply_upright_constraint), in which case `rotmat @ reference_rotmat.T` is
    -- up to floating point -- exactly a rotation about shared_up, so this
    decomposition is well-defined and lossless (apply_yaw_angle inverts it).
    """
    delta = rotmat @ reference_rotmat.T
    e1, e2 = perpendicular_basis(shared_up)
    rotated_e1 = delta @ e1
    return float(np.arctan2(np.dot(rotated_e1, e2), np.dot(rotated_e1, e1)))


def apply_yaw_angle(angle: float, shared_up: np.ndarray, reference_rotmat: np.ndarray) -> np.ndarray:
    """Inverse of extract_yaw_angle: rebuild the rotation matrix for a given yaw."""
    axis = shared_up / np.linalg.norm(shared_up)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float32,
    )
    rotation_about_axis = (
        np.eye(3, dtype=np.float32) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    ).astype(np.float32)
    return rotation_about_axis @ reference_rotmat


def weighted_median_per_dim(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-dimension weighted median of a (N, D) batch, weights shape (N,).

    Used to lock one shape_params/scale_params per character: a robust
    combination (unlike a weighted mean) so that a handful of badly-fit frames
    can't pull the character's locked body proportions away from the
    consensus of its better-fit frames.
    """
    result = np.zeros(values.shape[1], dtype=np.float32)
    for dim in range(values.shape[1]):
        order = np.argsort(values[:, dim])
        sorted_values = values[order, dim]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        cutoff = 0.5 * cumulative[-1]
        idx = int(np.searchsorted(cumulative, cutoff))
        idx = min(idx, len(sorted_values) - 1)
        result[dim] = sorted_values[idx]
    return result
