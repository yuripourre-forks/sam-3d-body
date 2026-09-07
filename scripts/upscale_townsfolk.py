#!/usr/bin/env python3
"""Batch-upscale split townsfolk frames, then reassemble the result for QA.

Supports two interchangeable upscaler backends (HYPIR and a lightweight
Clarity-style SD2.1 img2img proxy, see scripts/clarity_upscale.py) so their
output quality can be compared on the same frames.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.clarity_upscale import DEFAULT_BASE_MODEL as CLARITY_DEFAULT_BASE_MODEL
from scripts.clarity_upscale import ClarityStyleUpscaler
from scripts.hypir_upscale import DEFAULT_WEIGHT_PATH, HypirUpscaler
from scripts.reassemble_townsfolk import reassemble_from_manifest

UPSCALER_CHOICES = ("hypir", "clarity")


def build_upscaler(args):
    if args.upscaler == "hypir":
        return HypirUpscaler(
            weight_path=args.hypir_weight,
            base_model_path=args.hypir_base_model,
            device=args.device,
            upscale=args.upscale,
            prompt=args.prompt,
        )
    return ClarityStyleUpscaler(
        base_model_path=args.clarity_base_model,
        device=args.device,
        upscale=args.upscale,
        prompt=args.prompt,
        denoising_strength=args.clarity_denoise,
    )


def main():
    parser = argparse.ArgumentParser(description="Upscale townsfolk frames")
    parser.add_argument("--split-dir", default="output/split")
    parser.add_argument("--output-dir", default=None, help="Default: output/upscaled_<upscaler>")
    parser.add_argument("--upscaler", choices=UPSCALER_CHOICES, default="hypir")
    parser.add_argument("--hypir-weight", default=str(DEFAULT_WEIGHT_PATH))
    parser.add_argument(
        "--hypir-base-model", default=str(REPO_ROOT / "checkpoints/sd2-1-base")
    )
    parser.add_argument("--clarity-base-model", default=CLARITY_DEFAULT_BASE_MODEL)
    parser.add_argument("--clarity-denoise", type=float, default=0.35)
    parser.add_argument(
        "--prompt", default="isometric pixel art game character sprite, sharp details"
    )
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--char-start", type=int, default=0)
    parser.add_argument("--char-end", type=int, default=None)
    parser.add_argument(
        "--skip-reassemble",
        action="store_true",
        help="Skip writing a reassembled full sprite sheet for QA after upscaling",
    )
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir

    output_dir = args.output_dir or f"output/upscaled_{args.upscaler}"
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    with open(split_dir / "manifest.json", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    char_end = args.char_end if args.char_end is not None else len(manifest["characters"])

    upscaler = build_upscaler(args)
    upscaler.load()

    for char_entry in manifest["characters"][args.char_start : char_end]:
        char_id = char_entry["id"]
        for frame_info in char_entry["frames"]:
            frame_path = split_dir / frame_info["path"]
            output_path = output_dir / frame_info["path"]
            upscaler.upscale_file(frame_path, output_path)
            print(f"Upscaled {frame_path.relative_to(split_dir)}")

    print(f"Done. Output: {output_dir}")

    if not args.skip_reassemble:
        sheet_path = output_dir / "townsfolk_upscaled.png"
        reassemble_from_manifest(
            split_dir, sheet_path, frames_dir=output_dir, scale=args.upscale
        )
        print(f"Wrote QA sprite sheet: {sheet_path}")


if __name__ == "__main__":
    main()
