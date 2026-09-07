"""Foot grounding and isometric mesh rendering utilities."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pyrender
import trimesh

from sam_3d_body.visualization.renderer import Renderer, create_raymond_lights
from scripts.split_townsfolk import is_magenta_mask, is_white_mask

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

# MHR70 foot keypoint indices in pred_keypoints_3d
FOOT_KEYPOINT_INDICES = [15, 16, 17, 18, 19, 20]

ISO_ELEVATION_DEG = 30.0
ISO_AZIMUTH_DEG = 45.0
DEFAULT_RENDER_SIZE = 512
FILL_HEIGHT_RATIO = 0.8


def compute_ground_offset(keypoints_3d: np.ndarray) -> float:
    """Return Y translation so the lowest foot point touches the ground plane."""
    foot_points = keypoints_3d[FOOT_KEYPOINT_INDICES]
    return -float(np.min(foot_points[:, 1]))


def apply_ground_translation(
    vertices: np.ndarray,
    keypoints_3d: np.ndarray,
    joint_coords: np.ndarray | None = None,
    extra_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Translate mesh vertices, keypoints, and joints so feet rest on Y=0.

    cam_t is intentionally left untouched by callers: renderers place the mesh with
    a single `vertices + cam_t` addition (see Renderer.vertices_to_trimesh), so the
    ground offset must be baked into exactly one of the two operands, not both.
    A prior version of this function also translated cam_t, which silently doubled
    the vertical shift applied at render time -- for characters whose padding-heavy
    source frame already produces a large offset, that doubled shift pushed the
    mesh entirely outside the camera's view frustum (a blank mesh_render.png).
    """
    offset = compute_ground_offset(keypoints_3d) + extra_offset
    translation = np.array([0.0, offset, 0.0], dtype=np.float32)

    grounded_vertices = vertices + translation
    grounded_keypoints_3d = keypoints_3d + translation
    grounded_joints = None
    if joint_coords is not None:
        grounded_joints = joint_coords + translation

    return grounded_vertices, grounded_keypoints_3d, grounded_joints


def compute_character_ground_offset(
    frame_outputs: list[dict[str, Any]],
) -> float:
    """Use median foot height across frames to reduce per-frame bobbing."""
    offsets = []
    for frame_output in frame_outputs:
        keypoints_3d = frame_output["pred_keypoints_3d"]
        offsets.append(compute_ground_offset(keypoints_3d))
    return float(np.median(offsets))


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = map(math.radians, (rx_deg, ry_deg, rz_deg))
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)],
        ]
    )
    rot_y = np.array(
        [
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)],
        ]
    )
    rot_z = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1],
        ]
    )
    return rot_z @ rot_y @ rot_x


ISO_CAMERA_DISTANCE = 3.0
ISO_ORTHO_EXTENT = 1.2
WORLD_UP = (0.0, 1.0, 0.0)


def look_at_pose(
    eye: np.ndarray, target: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """Build a pyrender camera pose (camera-to-world) that looks from eye at target."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    # pyrender/OpenGL cameras look down their local -Z axis, so the pose's Z basis
    # vector must point from target back to eye (i.e. "backward").
    backward = eye - target
    backward /= np.linalg.norm(backward)
    up = np.asarray(WORLD_UP, dtype=np.float64)
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    true_up = np.cross(backward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def render_mesh_isometric(
    vertices: np.ndarray,
    faces: np.ndarray,
    render_size: int = DEFAULT_RENDER_SIZE,
    mesh_color: tuple[float, float, float] = LIGHT_BLUE,
    elevation_deg: float = ISO_ELEVATION_DEG,
    azimuth_deg: float = ISO_AZIMUTH_DEG,
    roll_deg: float = 0.0,
    extent_override: float | None = None,
    view_yaw_deg: float = 0.0,
) -> np.ndarray:
    """Render mesh-only RGBA image with an orthographic isometric orbit camera.

    `view_yaw_deg` adds a known per-frame camera orbit offset around world Y.
    This is equivalent to rotating the character by -view_yaw_deg with a fixed
    camera (character-rotates == camera-orbits for orthographic silhouettes).

    extent_override, when given, replaces the per-call `max ||v - mean||`
    normalization with a caller-supplied constant. This is what lets a shared,
    fixed metric-to-pixel scale be used across every frame of a character (see
    scripts/fit_shared_camera.py) instead of independently rescaling each frame
    to fill the canvas, which otherwise erases true relative body size and
    root motion (an outstretched arm would no longer shrink the whole figure).

    roll_deg applies an in-plane rotation of the rendered image about its own
    center -- equivalent to camera roll for an orthographic camera, and much
    simpler/safer to add this way than composing another mesh-space Euler
    rotation (which is exactly what previously introduced the "tipped over"
    roll artifact this renderer was rewritten to avoid).
    """
    centered_vertices = vertices - vertices.mean(axis=0)
    if extent_override is not None:
        extent = extent_override
    else:
        extent = np.max(np.linalg.norm(centered_vertices, axis=1))
    if extent < 1e-6:
        extent = 1.0
    normalized_vertices = centered_vertices / extent

    # SAM3D-Body vertices come out in a Y-down/Z-forward camera convention (the same
    # one render_mesh_only_perspective un-does via a 180-degree flip around X before
    # rendering). Apply that same flip first so the mesh is upright in render space.
    render_space_flip = rotation_matrix_xyz(180.0, 0.0, 0.0)
    upright_vertices = normalized_vertices @ render_space_flip.T

    # Only yaw the mesh around the world-up axis for azimuth. Elevation is handled by
    # orbiting the *camera* (via a look-at pose) rather than pitching the mesh with a
    # second Euler rotation: composing a pitch and a yaw directly on the vertices
    # introduces an unwanted in-plane roll of the character's up axis, which made
    # standing poses look tipped over into a bogus "diving forward" orientation.
    azimuth_rotation = rotation_matrix_xyz(0.0, azimuth_deg + view_yaw_deg, 0.0)
    mesh_vertices = upright_vertices @ azimuth_rotation.T

    vertex_colors = np.array(
        [int(c * 255) for c in mesh_color] + [255],
        dtype=np.uint8,
    )
    vertex_colors = np.tile(vertex_colors, (mesh_vertices.shape[0], 1))
    mesh = trimesh.Trimesh(
        mesh_vertices,
        faces.copy(),
        vertex_colors=vertex_colors,
        process=False,
    )

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.5, 0.5, 0.5))
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    camera = pyrender.OrthographicCamera(
        xmag=ISO_ORTHO_EXTENT,
        ymag=ISO_ORTHO_EXTENT,
    )
    elevation_rad = math.radians(elevation_deg)
    camera_eye = ISO_CAMERA_DISTANCE * np.array(
        [0.0, math.sin(elevation_rad), math.cos(elevation_rad)]
    )
    camera_pose = look_at_pose(camera_eye)
    scene.add(camera, pose=camera_pose)

    for light_node in create_raymond_lights():
        scene.add_node(light_node)

    renderer = pyrender.OffscreenRenderer(render_size, render_size)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    if roll_deg != 0.0:
        center = (render_size / 2.0, render_size / 2.0)
        rotation_2d = cv2.getRotationMatrix2D(center, roll_deg, 1.0)
        color = cv2.warpAffine(
            color,
            rotation_2d,
            (render_size, render_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    return color.astype(np.float32) / 255.0


def render_mesh_overlay(
    renderer: Renderer,
    vertices: np.ndarray,
    cam_t: np.ndarray,
    image_bgr: np.ndarray,
) -> np.ndarray:
    """Render perspective mesh overlay using the SAM3D renderer."""
    overlay = (
        renderer(
            vertices,
            cam_t,
            image_bgr.copy(),
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    ).astype(np.uint8)
    return overlay


def render_mesh_only_perspective(
    renderer: Renderer,
    vertices: np.ndarray,
    cam_t: np.ndarray,
    render_width: int,
    render_height: int,
    render_scale: float = 1.0,
) -> np.ndarray:
    """Render mesh-only RGBA with a perspective camera, at the source crop's aspect ratio.

    render_width/render_height should be proportional to the bbox SAM3D-Body was run
    on (e.g. that bbox size times render_scale). Rendering at a mismatched aspect
    ratio -- or scaling the resolution without scaling focal_length to match --
    shifts/squashes the mesh relative to where the source crop's own camera placed
    it, which breaks pixel-accurate reassembly against the original sprite sheet.
    """
    scaled_renderer = Renderer(
        focal_length=renderer.focal_length * render_scale, faces=renderer.faces
    )
    rgba = scaled_renderer.render_rgba(
        vertices,
        cam_t=cam_t,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(0, 0, 0),
        render_res=[render_width, render_height],
    )
    return rgba


KP2D_BBOX_MARGIN_RATIO = 1.3
FRONTAL_SUPERSAMPLE_FACTOR = 4
ALPHA_VISIBILITY_THRESHOLD = 0.01


class ContentBBoxAndCentroid:
    """Tight bbox + mass-centroid of a mask's foreground pixels."""

    def __init__(self, bbox: tuple[int, int, int, int], centroid: tuple[float, float]):
        self.bbox = bbox
        self.centroid = centroid


def content_bbox_and_centroid_from_mask(mask: np.ndarray) -> ContentBBoxAndCentroid | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    bbox = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    centroid = (float(cols.mean()), float(rows.mean()))
    return ContentBBoxAndCentroid(bbox, centroid)


def scale_content_target(content: ContentBBoxAndCentroid, factor: float) -> ContentBBoxAndCentroid:
    """Scale a bbox+centroid by `factor` (e.g. to go from 1x sprite to upscaled canvas coords)."""
    x0, y0, x1, y1 = content.bbox
    centroid_x, centroid_y = content.centroid
    return ContentBBoxAndCentroid(
        bbox=(x0 * factor, y0 * factor, x1 * factor, y1 * factor),
        centroid=(centroid_x * factor, centroid_y * factor),
    )


def compute_frame_content_bbox_and_centroid(frame_path: str) -> ContentBBoxAndCentroid | None:
    """Tight bbox + mass-centroid of a split sprite frame's own character pixels.

    This is ground truth for where/how-large the character is within its frame
    (it comes directly from the original artwork), so it makes a much more
    accurate render_mesh_frontal_placed target than any model-derived estimate.
    The centroid (rather than just the bbox center) matters for asymmetric poses
    -- e.g. one arm held out to the side shifts the bbox center away from where
    the body's visual mass actually is, which showed up as a consistent
    left/right placement bias for such frames when bbox-center alignment was
    used. Returns None if the frame has no non-background content.
    """
    image_bgra = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if image_bgra is None:
        raise ValueError(f"Could not load frame: {frame_path}")

    rgb = cv2.cvtColor(image_bgra[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.int32)
    if image_bgra.shape[2] == 4:
        alpha = image_bgra[:, :, 3]
    else:
        alpha = np.full(image_bgra.shape[:2], 255, dtype=np.uint8)

    content_mask = ~is_magenta_mask(rgb) & ~is_white_mask(rgb) & (alpha > 0)
    return content_bbox_and_centroid_from_mask(content_mask)


def keypoints_2d_target(keypoints_2d: np.ndarray) -> ContentBBoxAndCentroid:
    """Fallback placement target derived from pred_keypoints_2d's own extent.

    Only used when the source sprite frame's silhouette bbox isn't available.
    pred_keypoints_2d is always trustworthy (it's the model's actual 2D output,
    so it's within the source crop by construction), but ankle/head-top joints
    sit slightly inside the true body silhouette, so the raw keypoint bbox is
    expanded by KP2D_BBOX_MARGIN_RATIO to approximate the full body extent. The
    keypoint centroid is used as-is (it isn't affected by that margin).
    """
    kp_x0, kp_y0 = keypoints_2d[:, 0].min(), keypoints_2d[:, 1].min()
    kp_x1, kp_y1 = keypoints_2d[:, 0].max(), keypoints_2d[:, 1].max()
    center_x, center_y = (kp_x0 + kp_x1) / 2.0, (kp_y0 + kp_y1) / 2.0
    half_width = (kp_x1 - kp_x0) * KP2D_BBOX_MARGIN_RATIO / 2.0
    half_height = (kp_y1 - kp_y0) * KP2D_BBOX_MARGIN_RATIO / 2.0
    bbox = (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )
    centroid = (float(keypoints_2d[:, 0].mean()), float(keypoints_2d[:, 1].mean()))
    return ContentBBoxAndCentroid(bbox, centroid)


def render_mesh_frontal_placed(
    vertices: np.ndarray,
    faces: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    target: ContentBBoxAndCentroid | None = None,
    keypoints_2d: np.ndarray | None = None,
    elevation_deg: float = 0.0,
    azimuth_deg: float = 0.0,
    roll_deg: float = 0.0,
    extent_override: float | None = None,
) -> np.ndarray:
    """Render the mesh from a given camera and place it to match target's extent/centroid.

    elevation_deg/azimuth_deg/roll_deg/extent_override default to the original
    fixed frontal camera + per-frame extent normalization; scripts/
    fit_shared_camera.py and the later shared-camera pipeline stages pass in
    the fitted shared camera and a per-character extent reference instead, so
    every frame of the sheet is rendered with one consistent camera/scale.

    SAM3D-Body's own weak-perspective camera (pred_cam_t/focal_length) is fit to
    reproduce pred_keypoints_2d, but reprojecting pred_vertices with that same
    camera does not reliably reproduce pred_keypoints_2d for crops that are mostly
    padding (verified empirically: the two disagree by a large, character-dependent
    vertical bias for such frames), which can push the reprojected mesh entirely
    outside the render canvas. So instead of re-deriving that camera, we render the
    mesh in a fixed, self-centered frontal view (immune to any camera
    mis-projection) and place/scale that render to cover target's bbox -- ideally
    from compute_frame_content_bbox_and_centroid() (ground truth for
    where/how-large the character is), falling back to keypoints_2d_target() when
    unavailable. Placement aligns content *centroids* (not bbox centers) on both
    sides: for asymmetric poses (e.g. one arm held out to the side), the bbox
    center sits away from the body's visual mass, which otherwise shows up as a
    consistent left/right offset between the render and the original sprite.
    """
    if target is None:
        if keypoints_2d is None:
            raise ValueError("render_mesh_frontal_placed requires target or keypoints_2d")
        target = keypoints_2d_target(keypoints_2d)

    render_size = max(canvas_width, canvas_height) * FRONTAL_SUPERSAMPLE_FACTOR
    frontal = render_mesh_isometric(
        vertices,
        faces,
        render_size=render_size,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        roll_deg=roll_deg,
        extent_override=extent_override,
    )

    alpha = frontal[:, :, 3]
    frontal_content = content_bbox_and_centroid_from_mask(alpha > ALPHA_VISIBILITY_THRESHOLD)
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    if frontal_content is None:
        return canvas

    col_min, row_min, col_max, row_max = frontal_content.bbox
    cropped = frontal[row_min:row_max, col_min:col_max]
    cropped_height, cropped_width = cropped.shape[:2]
    crop_centroid_x = frontal_content.centroid[0] - col_min
    crop_centroid_y = frontal_content.centroid[1] - row_min

    target_x0, target_y0, target_x1, target_y1 = target.bbox
    target_width = target_x1 - target_x0
    target_height = target_y1 - target_y0
    target_centroid_x, target_centroid_y = target.centroid

    scale = min(target_width / cropped_width, target_height / cropped_height)
    resized_width = max(1, int(round(cropped_width * scale)))
    resized_height = max(1, int(round(cropped_height * scale)))
    cropped_uint8 = np.clip(cropped * 255, 0, 255).astype(np.uint8)
    resized = cv2.resize(
        cropped_uint8, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    resized_centroid_x = crop_centroid_x * scale
    resized_centroid_y = crop_centroid_y * scale

    paste_x0 = int(round(target_centroid_x - resized_centroid_x))
    paste_y0 = int(round(target_centroid_y - resized_centroid_y))
    src_x0, src_y0 = max(0, -paste_x0), max(0, -paste_y0)
    dst_x0, dst_y0 = max(0, paste_x0), max(0, paste_y0)
    dst_x1 = min(canvas_width, paste_x0 + resized_width)
    dst_y1 = min(canvas_height, paste_y0 + resized_height)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[
            src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)
        ]
    return canvas


def render_mesh_shared_placed(
    vertices: np.ndarray,
    faces: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    elevation_deg: float,
    azimuth_deg: float,
    roll_deg: float,
    extent_override: float,
    target_centroid_x: float,
    target_floor_row: float,
    view_yaw_deg: float = 0.0,
    angle_nudge_deg: float = 0.0,
) -> np.ndarray:
    """Stage 6 placement: shared camera + shared per-character scale + shared floor row.

    Unlike render_mesh_frontal_placed (which independently rescales *and*
    recenters every frame to exactly cover that frame's own sprite content
    bbox -- appropriate before real grounding existed, but destructive of true
    relative body size and vertical foot motion), this renders with one fixed
    camera and one fixed per-character extent_override (both from
    scripts/fit_shared_camera.py's output/camera.json), so a frame's rendered
    size only changes when the pose itself changes (e.g. crouching), not
    frame-to-frame arbitrarily.

    Placement: horizontal position tracks the sprite's own per-frame content
    centroid (preserving genuine horizontal sway/motion, still grounded in the
    original artwork), while vertical position anchors the render's own lowest
    opaque pixel to target_floor_row -- one shared "floor line" per character
    (see compute_character_floor_row) -- so real foot height, bob, and ground
    contact survive instead of being erased by per-frame recentering.
    """
    render_size = max(canvas_width, canvas_height) * FRONTAL_SUPERSAMPLE_FACTOR
    frontal = render_mesh_isometric(
        vertices,
        faces,
        render_size=render_size,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        roll_deg=roll_deg,
        extent_override=extent_override,
        view_yaw_deg=view_yaw_deg + angle_nudge_deg,
    )

    alpha = frontal[:, :, 3]
    frontal_content = content_bbox_and_centroid_from_mask(alpha > ALPHA_VISIBILITY_THRESHOLD)
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    if frontal_content is None:
        return canvas

    col_min, row_min, col_max, row_max = frontal_content.bbox
    cropped = frontal[row_min:row_max, col_min:col_max]
    cropped_uint8 = np.clip(cropped * 255, 0, 255).astype(np.uint8)

    downscale = 1.0 / FRONTAL_SUPERSAMPLE_FACTOR
    resized_width = max(1, int(round(cropped.shape[1] * downscale)))
    resized_height = max(1, int(round(cropped.shape[0] * downscale)))
    resized = cv2.resize(cropped_uint8, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    crop_centroid_x = (frontal_content.centroid[0] - col_min) * downscale
    paste_x0 = int(round(target_centroid_x - crop_centroid_x))
    # The crop is tight to the alpha mask, so its own last row is the mesh's lowest
    # rendered pixel -- placing that row at target_floor_row is exactly "feet on floor".
    paste_y0 = int(round(target_floor_row - (resized_height - 1)))

    src_x0, src_y0 = max(0, -paste_x0), max(0, -paste_y0)
    dst_x0, dst_y0 = max(0, paste_x0), max(0, paste_y0)
    dst_x1 = min(canvas_width, paste_x0 + resized_width)
    dst_y1 = min(canvas_height, paste_y0 + resized_height)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[
            src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)
        ]
    return canvas


def render_mesh_centroid_placed(
    vertices: np.ndarray,
    faces: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    elevation_deg: float,
    azimuth_deg: float,
    roll_deg: float,
    extent_override: float,
    target_centroid_x: float,
    target_centroid_y: float,
    view_yaw_deg: float = 0.0,
    angle_nudge_deg: float = 0.0,
) -> np.ndarray:
    """Render with a fixed camera/extent and place by translation only (no rescale).

    render_mesh_frontal_placed always rescales its render to exactly cover
    `target`'s bbox, which makes it useless for *fitting* extent_override/scale
    (whatever scale is passed in gets normalized away by that rescale, so the
    fitted value ends up driven by incidental clipping/resampling artifacts
    instead of genuine on-screen size -- see scripts/fit_shared_camera.py's
    per-character scale search, which must use this function instead). This
    mirrors render_mesh_shared_placed's non-rescaling placement (used at
    actual render time in scripts/render_final_poses.py) but anchors both
    axes to a target centroid instead of a shared floor row, since a
    lightweight camera/scale fit has no grounding/floor-row machinery set up.
    """
    render_size = max(canvas_width, canvas_height) * FRONTAL_SUPERSAMPLE_FACTOR
    frontal = render_mesh_isometric(
        vertices,
        faces,
        render_size=render_size,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        roll_deg=roll_deg,
        extent_override=extent_override,
        view_yaw_deg=view_yaw_deg + angle_nudge_deg,
    )

    alpha = frontal[:, :, 3]
    frontal_content = content_bbox_and_centroid_from_mask(alpha > ALPHA_VISIBILITY_THRESHOLD)
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    if frontal_content is None:
        return canvas

    col_min, row_min, col_max, row_max = frontal_content.bbox
    cropped = frontal[row_min:row_max, col_min:col_max]
    cropped_uint8 = np.clip(cropped * 255, 0, 255).astype(np.uint8)

    downscale = 1.0 / FRONTAL_SUPERSAMPLE_FACTOR
    resized_width = max(1, int(round(cropped.shape[1] * downscale)))
    resized_height = max(1, int(round(cropped.shape[0] * downscale)))
    resized = cv2.resize(cropped_uint8, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    crop_centroid_x = (frontal_content.centroid[0] - col_min) * downscale
    crop_centroid_y = (frontal_content.centroid[1] - row_min) * downscale
    paste_x0 = int(round(target_centroid_x - crop_centroid_x))
    paste_y0 = int(round(target_centroid_y - crop_centroid_y))

    src_x0, src_y0 = max(0, -paste_x0), max(0, -paste_y0)
    dst_x0, dst_y0 = max(0, paste_x0), max(0, paste_y0)
    dst_x1 = min(canvas_width, paste_x0 + resized_width)
    dst_y1 = min(canvas_height, paste_y0 + resized_height)

    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[
            src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)
        ]
    return canvas


def compute_character_floor_row(sprite_masks: list[np.ndarray]) -> float:
    """Median lowest-content-row across a character's own sprite frames.

    This is the shared "floor line" (in canvas pixel coordinates) that
    render_mesh_shared_placed anchors every frame's rendered feet to -- using
    all frames (not just a handful) and the sprite's own artwork (not the
    model's estimate) so a single mis-estimated frame can't shift the floor.
    """
    bottom_rows = []
    for mask in sprite_masks:
        rows = np.where(mask.any(axis=1))[0]
        if rows.size:
            bottom_rows.append(int(rows.max()))
    return float(np.median(bottom_rows)) if bottom_rows else 0.0


def save_rgba_image(rgba: np.ndarray, output_path: str) -> None:
    """Save float RGBA array as PNG."""
    if rgba.dtype != np.uint8:
        rgba_uint8 = np.clip(rgba * 255, 0, 255).astype(np.uint8)
    else:
        rgba_uint8 = rgba
    cv2.imwrite(output_path, cv2.cvtColor(rgba_uint8, cv2.COLOR_RGBA2BGRA))


def assemble_sprite_sheet(
    frame_images: list[np.ndarray],
    output_path: str,
) -> None:
    """Concatenate frame images horizontally into a sprite sheet."""
    if not frame_images:
        return

    heights = [img.shape[0] for img in frame_images]
    widths = [img.shape[1] for img in frame_images]
    sheet_height = max(heights)
    sheet_width = sum(widths)

    if frame_images[0].dtype != np.uint8:
        normalized = [
            np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else img
            for img in frame_images
        ]
    else:
        normalized = frame_images

    channels = normalized[0].shape[2] if len(normalized[0].shape) == 3 else 1
    if channels == 4:
        sheet = np.zeros((sheet_height, sheet_width, 4), dtype=np.uint8)
    else:
        sheet = np.zeros((sheet_height, sheet_width, 3), dtype=np.uint8)

    x_offset = 0
    for img in normalized:
        h, w = img.shape[:2]
        y_offset = (sheet_height - h) // 2
        if channels == 4:
            sheet[y_offset : y_offset + h, x_offset : x_offset + w] = img
        else:
            sheet[y_offset : y_offset + h, x_offset : x_offset + w] = img
        x_offset += w

    if channels == 4:
        cv2.imwrite(output_path, cv2.cvtColor(sheet, cv2.COLOR_RGBA2BGRA))
    else:
        cv2.imwrite(output_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
