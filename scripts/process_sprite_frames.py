#!/usr/bin/env python3
"""Process sprite frames with SAM3D-Body using sheet config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.process_townsfolk import main as process_main
from scripts.sprite_config import load_sheet_config

CANDIDATE_UPSCALE = {
    "hypir4x": 4,
    "hypir8x": 8,
    "lanczos4x": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Process sprite frames with SAM3D-Body")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--upscale", type=int, default=None)
    parser.add_argument("--render-scale", type=int, default=None)
    parser.add_argument(
        "--candidate",
        default=None,
        help="Upscale candidate name (hypir4x, hypir8x, lanczos4x)",
    )
    args, remaining = parser.parse_known_args()

    if args.config:
        config = load_sheet_config(args.config)
        split_dir = args.split_dir or str(config.split_dir)
        output_dir = args.output_dir or str(config.output_root / "inference")
        device = args.device or config.device
        candidate = args.candidate or (
            config.candidate_upscalers[0] if config.candidate_upscalers else None
        )
        upscale = args.upscale or CANDIDATE_UPSCALE.get(candidate or "", config.upscale)
        render_scale = args.render_scale or config.render_scale
    else:
        split_dir = args.split_dir or "output/split"
        output_dir = args.output_dir or "output/inference"
        device = args.device or "cuda"
        candidate = args.candidate
        upscale = args.upscale or 4
        render_scale = args.render_scale or 4

    argv = [
        "process_sprite_frames.py",
        "--split-dir",
        split_dir,
        "--output-dir",
        output_dir,
        "--device",
        device,
        "--upscale",
        str(upscale),
        "--render-scale",
        str(render_scale),
    ]
    if candidate == "lanczos4x":
        argv.append("--no-hypir")
    elif candidate == "hypir8x":
        argv.extend(["--upscale", "8"])
    argv.extend(remaining)
    sys.argv = argv
    process_main()


if __name__ == "__main__":
    main()
