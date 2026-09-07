#!/usr/bin/env python3
"""Shared sprite-sheet grid utilities: background masks, white-line detection,
fixed-pitch cell counting, and square frame normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

MAGENTA_RGBA = np.array([255, 0, 255, 255], dtype=np.uint8)
MAGENTA_LOW = np.array([180, 0, 0], dtype=np.int32)
MAGENTA_HIGH = np.array([255, 120, 255], dtype=np.int32)
DEFAULT_WHITE_THRESHOLD = 240
MIN_MAGENTA_RATIO = 0.05
MIN_CONTENT_BAND_SIZE = 8


def is_magenta_mask(rgb: np.ndarray, white_threshold: int = DEFAULT_WHITE_THRESHOLD) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > MAGENTA_LOW[0]) & (g < MAGENTA_HIGH[1]) & (b > MAGENTA_LOW[2])


def is_white_mask(rgb: np.ndarray, white_threshold: int = DEFAULT_WHITE_THRESHOLD) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > white_threshold) & (g > white_threshold) & (b > white_threshold)


def content_mask(
    rgba: np.ndarray,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    ignore_white: bool = True,
) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.int32)
    alpha = rgba[..., 3] if rgba.shape[2] == 4 else np.full(rgba.shape[:2], 255)
    white = is_white_mask(rgb, white_threshold) if ignore_white else np.zeros(rgba.shape[:2], dtype=bool)
    return ~is_magenta_mask(rgb) & ~white & (alpha > 0)


def find_separator_bands(
    fraction_along_axis: np.ndarray,
    threshold: float,
    min_band_size: int = 1,
) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for index, value in enumerate(fraction_along_axis):
        is_separator = value >= threshold
        if is_separator and not in_band:
            start = index
            in_band = True
        elif not is_separator and in_band:
            if index - start >= min_band_size:
                bands.append((start, index - 1))
            in_band = False
    if in_band and len(fraction_along_axis) - start >= min_band_size:
        bands.append((start, len(fraction_along_axis) - 1))
    return bands


def content_regions_between_separators(
    length: int,
    separator_bands: list[tuple[int, int]],
    min_region_size: int = MIN_CONTENT_BAND_SIZE,
) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    cursor = 0
    for sep_start, sep_end in separator_bands:
        if sep_start > cursor:
            region_end = sep_start - 1
            if region_end - cursor + 1 >= min_region_size:
                regions.append((cursor, region_end))
        cursor = sep_end + 1
    if length - cursor >= min_region_size:
        regions.append((cursor, length - 1))
    return regions


def count_used_cells(
    rgba: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    num_cols: int,
    cell_size: int,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    min_magenta_ratio: float = MIN_MAGENTA_RATIO,
) -> int:
    sheet_height, sheet_width = rgba.shape[:2]
    row_bottom = min(y1 + 1, sheet_height)
    row_top = max(y0, 0)
    if row_top >= sheet_height:
        return 0

    row_rgb = rgba[row_top:row_bottom, :, :3].astype(np.int32)
    magenta_row = is_magenta_mask(row_rgb, white_threshold)
    min_magenta_pixels = min_magenta_ratio * cell_size * cell_size
    last_used = -1

    for col_index in range(num_cols):
        cell_x0 = x0 + col_index * cell_size
        cell_x1 = min(cell_x0 + cell_size, sheet_width)
        if cell_x0 >= sheet_width:
            break
        slot = magenta_row[:, cell_x0:cell_x1]
        if slot.size == 0:
            continue
        if slot.sum() >= min_magenta_pixels:
            last_used = col_index
    return last_used + 1 if last_used >= 0 else 0


def tight_content_bbox(
    rgba: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
) -> tuple[int, int, int, int] | None:
    crop = rgba[y0:y1, x0:x1]
    mask = content_mask(crop, white_threshold=white_threshold)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (
        int(x0 + xs.min()),
        int(y0 + ys.min()),
        int(x0 + xs.max() + 1),
        int(y0 + ys.max() + 1),
    )


@dataclass
class NormalizedFrame:
    rgba: np.ndarray
    normalize_scale: float
    content_bbox: tuple[int, int, int, int]
    paste_offset: tuple[int, int]


def normalize_to_square(
    rgba: np.ndarray,
    output_size: int,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    background_rgba: tuple[int, int, int, int] = (255, 0, 255, 255),
) -> NormalizedFrame | None:
    mask = content_mask(rgba, white_threshold=white_threshold)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    content = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    content_h, content_w = content.shape[:2]
    scale = output_size / max(content_h, content_w)
    new_w = max(1, int(round(content_w * scale)))
    new_h = max(1, int(round(content_h * scale)))
    resized = np.array(Image.fromarray(content).resize((new_w, new_h), Image.NEAREST))
    canvas = np.tile(np.array(background_rgba, dtype=np.uint8), (output_size, output_size, 1))
    offset_x = (output_size - new_w) // 2
    offset_y = (output_size - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return NormalizedFrame(
        rgba=canvas,
        normalize_scale=float(scale),
        content_bbox=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
        paste_offset=(offset_x, offset_y),
    )


def denormalized_placement_geometry(
    frame_info: dict,
    output_frame_size: int,
    render_scale: int,
) -> tuple[int, int, int, int, int, int] | None:
    """Geometry to invert normalize_to_square()'s centered scale-to-fit transform.

    normalize_to_square() takes a frame's tight local content bbox
    (frame_info["source_bbox"], in original-sheet pixel units local to the
    frame's own grid cell -- NOT full-sheet coordinates) and scales+centers it
    into a fixed `output_frame_size` square using `frame_info["normalize_scale"]`.
    Any render generated against that normalized square (e.g. mesh_render.png,
    itself `output_frame_size * render_scale` pixels) must be unwound through
    the same transform, then repositioned at the frame's own location in the
    full original sheet (frame_info["grid_cell_bbox"]), in order to align with
    the un-normalized original sheet for reassembly/comparison. The centering
    formula here must stay in sync with normalize_to_square()'s own
    `offset_x = (output_size - new_w) // 2` (and offset_y equivalent).

    Returns (crop_x0, crop_y0, crop_x1, crop_y1, paste_x, paste_y) in
    render_scale-upscaled pixel units, or None if the frame lacks the
    necessary manifest fields or has degenerate content.
    """
    source_bbox = frame_info.get("source_bbox")
    grid_cell_bbox = frame_info.get("grid_cell_bbox")
    normalize_scale = frame_info.get("normalize_scale")
    if not source_bbox or not grid_cell_bbox or not normalize_scale:
        return None
    content_width = source_bbox[2] - source_bbox[0]
    content_height = source_bbox[3] - source_bbox[1]
    if content_width <= 0 or content_height <= 0:
        return None
    normalized_width = max(1, int(round(content_width * normalize_scale)))
    normalized_height = max(1, int(round(content_height * normalize_scale)))
    normalized_offset_x = (output_frame_size - normalized_width) // 2
    normalized_offset_y = (output_frame_size - normalized_height) // 2
    crop_x0 = normalized_offset_x * render_scale
    crop_y0 = normalized_offset_y * render_scale
    crop_x1 = crop_x0 + normalized_width * render_scale
    crop_y1 = crop_y0 + normalized_height * render_scale
    paste_x = (grid_cell_bbox[0] + source_bbox[0]) * render_scale
    paste_y = (grid_cell_bbox[1] + source_bbox[1]) * render_scale
    return crop_x0, crop_y0, crop_x1, crop_y1, paste_x, paste_y


def denormalize_render(
    square_rgba: np.ndarray,
    frame_info: dict,
    output_frame_size: int,
    render_scale: int,
) -> np.ndarray | None:
    """Crop+resize a normalized-square render back to the frame's own original
    (un-normalized) content footprint (ignoring where that footprint sits in
    the full sheet -- see place_normalized_render_on_sheet() for that).

    `square_rgba` is expected to be an `output_frame_size * render_scale`
    square render (e.g. mesh_render.png) generated against the normalized
    split frame. Returns None if placement geometry can't be computed or the
    crop is degenerate.
    """
    geometry = denormalized_placement_geometry(frame_info, output_frame_size, render_scale)
    if geometry is None:
        return None
    crop_x0, crop_y0, crop_x1, crop_y1, _paste_x, _paste_y = geometry
    crop = square_rgba[crop_y0:crop_y1, crop_x0:crop_x1]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
    source_bbox = frame_info["source_bbox"]
    target_width = (source_bbox[2] - source_bbox[0]) * render_scale
    target_height = (source_bbox[3] - source_bbox[1]) * render_scale
    return np.array(Image.fromarray(crop).resize((target_width, target_height), Image.LANCZOS))


def place_normalized_render_on_sheet(
    square_rgba: np.ndarray,
    frame_info: dict,
    output_frame_size: int,
    render_scale: int,
) -> tuple[np.ndarray, int, int] | None:
    """Crop+resize a normalized-square render back to its original-sheet footprint.

    Returns (resized_rgba, paste_x, paste_y) -- the content-only crop resized
    to match the frame's own original (un-normalized) footprint, and the
    top-left position at which to paste it into a full-sheet canvas sized
    `manifest["image_size"] * render_scale` -- or None if placement geometry
    can't be computed.
    """
    geometry = denormalized_placement_geometry(frame_info, output_frame_size, render_scale)
    if geometry is None:
        return None
    resized = denormalize_render(square_rgba, frame_info, output_frame_size, render_scale)
    if resized is None:
        return None
    _, _, _, _, paste_x, paste_y = geometry
    return resized, paste_x, paste_y


def extract_cell(
    rgba: np.ndarray,
    x: int,
    y: int,
    cell_size: int,
) -> np.ndarray:
    sheet_height, sheet_width = rgba.shape[:2]
    frame = np.tile(MAGENTA_RGBA, (cell_size, cell_size, 1))
    src_x1 = min(x + cell_size, sheet_width)
    src_y1 = min(y + cell_size, sheet_height)
    if x < sheet_width and y < sheet_height:
        crop = rgba[y:src_y1, x:src_x1]
        frame[: crop.shape[0], : crop.shape[1]] = crop
    return frame


def draw_grid_preview(
    rgba: np.ndarray,
    blocks: list[dict],
    output_path: str,
) -> None:
    image = Image.fromarray(rgba.copy())
    draw = ImageDraw.Draw(image)
    colors = [
        (255, 64, 64, 255),
        (64, 255, 64, 255),
        (64, 128, 255, 255),
        (255, 200, 64, 255),
        (200, 64, 255, 255),
    ]
    for block_index, block in enumerate(blocks):
        color = colors[block_index % len(colors)]
        for stripe in block["stripes"]:
            y0, y1 = stripe["y0"], stripe["y1"]
            x0 = stripe["x0"]
            cell_size = stripe["cell_size"]
            num_cols = stripe["num_cols"]
            draw.rectangle((x0, y0, x0 + num_cols * cell_size - 1, y1), outline=color[:3], width=2)
            for col in range(num_cols):
                cx = x0 + col * cell_size
                draw.rectangle((cx, y0, cx + cell_size - 1, y1), outline=color[:3], width=1)
    image.save(output_path)
