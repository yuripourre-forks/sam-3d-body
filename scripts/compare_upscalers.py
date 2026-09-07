#!/usr/bin/env python3
"""Compare HYPIR vs. the lightweight Clarity-style proxy on the same sample frames.

Produces, per sample frame:
  - a side-by-side PNG (Lanczos baseline | HYPIR | Clarity-style)
  - a Laplacian-variance sharpness score for each variant (a standard, reference-free
    proxy for detail/edge strength -- higher generally means crisper output, but it
    is not a full perceptual quality metric, so the side-by-side images should be the
    primary basis for judging which upscaler to use).

Sharpness alone can't distinguish "crisp detail" from "noise/artifacts", so always
look at the side-by-side grid before picking a winner.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.clarity_upscale import ClarityStyleUpscaler
from scripts.hypir_upscale import DEFAULT_WEIGHT_PATH, HypirUpscaler

LABEL_BAR_HEIGHT = 24
LABEL_COLOR_BGR = (255, 255, 255)
LABEL_BG_BGR = (30, 30, 30)


def lanczos_upscale(bgr: np.ndarray, factor: int) -> np.ndarray:
    height, width = bgr.shape[:2]
    return cv2.resize(bgr, (width * factor, height * factor), interpolation=cv2.INTER_LANCZOS4)


def sharpness_score(bgr: np.ndarray) -> float:
    """Variance of the Laplacian: a common reference-free sharpness proxy."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def label_panel(image_bgr: np.ndarray, text: str) -> np.ndarray:
    width = image_bgr.shape[1]
    bar = np.full((LABEL_BAR_HEIGHT, width, 3), LABEL_BG_BGR, dtype=np.uint8)
    cv2.putText(
        bar, text, (4, LABEL_BAR_HEIGHT - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL_COLOR_BGR, 1
    )
    return np.vstack([bar, image_bgr])


def build_comparison_grid(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    labeled = [label_panel(img, text) for text, img in panels]
    max_height = max(p.shape[0] for p in labeled)
    padded = []
    for panel in labeled:
        pad_h = max_height - panel.shape[0]
        if pad_h > 0:
            panel = cv2.copyMakeBorder(panel, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded.append(panel)
    separator = np.full((max_height, 4, 3), (0, 0, 0), dtype=np.uint8)
    row = padded[0]
    for panel in padded[1:]:
        row = np.hstack([row, separator, panel])
    return row


def load_sample_frame_bgr(split_dir: Path, char_id: int, frame_idx: int) -> np.ndarray:
    frame_path = split_dir / f"char_{char_id:02d}" / f"frame_{frame_idx:03d}.png"
    rgba = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise ValueError(f"Could not load frame: {frame_path}")
    if rgba.shape[2] == 4:
        bgr = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3]
        neutral = np.full_like(bgr, 128, dtype=np.uint8)
        alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
        bgr = (bgr.astype(np.float32) * alpha_f + neutral.astype(np.float32) * (1 - alpha_f)).astype(
            np.uint8
        )
    else:
        bgr = rgba
    magenta_mask = (bgr[:, :, 2] > 200) & (bgr[:, :, 1] < 100) & (bgr[:, :, 0] > 200)
    bgr[magenta_mask] = 128
    return bgr


def parse_sample(spec: str) -> tuple[int, int]:
    char_str, frame_str = spec.split(":")
    return int(char_str), int(frame_str)


def main():
    parser = argparse.ArgumentParser(description="Compare HYPIR vs Clarity-style upscaler")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--output-dir", default="output/upscaler_comparison")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["0:0", "2:0", "3:0", "5:5", "7:2"],
        help="char_id:frame_idx pairs to compare, e.g. 0:0 3:5",
    )
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hypir-weight", default=str(DEFAULT_WEIGHT_PATH))
    parser.add_argument("--hypir-base-model", default=str(REPO_ROOT / "checkpoints/sd2-1-base"))
    parser.add_argument("--clarity-base-model", default=str(REPO_ROOT / "checkpoints/sd2-1-base"))
    parser.add_argument("--clarity-denoise", type=float, default=0.35)
    parser.add_argument(
        "--prompt", default="isometric pixel art game character sprite, sharp details"
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = [parse_sample(spec) for spec in args.samples]

    hypir = HypirUpscaler(
        weight_path=args.hypir_weight,
        base_model_path=args.hypir_base_model,
        device=args.device,
        upscale=args.upscale,
        prompt=args.prompt,
    )
    clarity = ClarityStyleUpscaler(
        base_model_path=args.clarity_base_model,
        device=args.device,
        upscale=args.upscale,
        prompt=args.prompt,
        denoising_strength=args.clarity_denoise,
    )

    results = []
    for char_id, frame_idx in samples:
        sample_name = f"char_{char_id:02d}_frame_{frame_idx:03d}"
        print(f"Comparing {sample_name}...")
        source_bgr = load_sample_frame_bgr(split_dir, char_id, frame_idx)

        lanczos_bgr = lanczos_upscale(source_bgr, args.upscale)
        hypir_bgr = hypir.upscale_bgr(source_bgr)
        clarity_bgr = clarity.upscale_bgr(source_bgr)

        target_size = (lanczos_bgr.shape[1], lanczos_bgr.shape[0])
        hypir_resized = cv2.resize(hypir_bgr, target_size, interpolation=cv2.INTER_NEAREST)
        clarity_resized = cv2.resize(clarity_bgr, target_size, interpolation=cv2.INTER_NEAREST)

        scores = {
            "lanczos": sharpness_score(lanczos_bgr),
            "hypir": sharpness_score(hypir_bgr),
            "clarity": sharpness_score(clarity_bgr),
        }
        print(
            f"  sharpness (Laplacian variance) -- lanczos: {scores['lanczos']:.1f}, "
            f"hypir: {scores['hypir']:.1f}, clarity: {scores['clarity']:.1f}"
        )

        sample_dir = output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / "lanczos.png"), lanczos_bgr)
        cv2.imwrite(str(sample_dir / "hypir.png"), hypir_bgr)
        cv2.imwrite(str(sample_dir / "clarity.png"), clarity_bgr)

        grid = build_comparison_grid(
            [
                (f"lanczos ({scores['lanczos']:.0f})", lanczos_bgr),
                (f"hypir ({scores['hypir']:.0f})", hypir_resized),
                (f"clarity ({scores['clarity']:.0f})", clarity_resized),
            ]
        )
        grid_path = sample_dir / "comparison.png"
        cv2.imwrite(str(grid_path), grid)

        results.append({"sample": sample_name, "scores": scores, "grid": str(grid_path)})

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(results, summary_file, indent=2)
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
