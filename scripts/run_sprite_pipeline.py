#!/usr/bin/env python3
"""Config-driven orchestrator for the generalized sprite-sheet pipeline."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.manifest_utils import load_manifest
from scripts.sprite_config import load_sheet_config

STAGES = [
    "split",
    "inference",
    "select",
    "multiview",
    "repose",
    "smooth",
    "camera",
    "render",
    "export",
]


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def sync_inference_dir(source_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)


def effective_candidate_upscalers(config, manifest: dict | None = None) -> list[str]:
    """Use a single upscaler for large multi-angle sheets to save inference cost."""
    if manifest is None:
        return list(config.candidate_upscalers)
    total_frames = sum(len(char_entry["frames"]) for char_entry in manifest["characters"])
    if total_frames > config.large_sheet_frame_threshold and len(config.candidate_upscalers) > 1:
        return [config.candidate_upscalers[0]]
    return list(config.candidate_upscalers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the generalized sprite pipeline from config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--from-stage", default="split", choices=STAGES)
    parser.add_argument("--to-stage", default="export", choices=STAGES)
    parser.add_argument("--dry-run-split", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    config = load_sheet_config(config_path)
    if args.device:
        config.device = args.device

    split_dir = config.split_dir
    output_root = config.output_root
    inference_dir = output_root / "inference"
    inference_dp = output_root / "inference_dp"
    inference_multiview = output_root / "inference_multiview"
    inference_reposed = output_root / "inference_reposed"
    inference_smoothed = output_root / "inference_smoothed"
    inference_final = output_root / "inference_final"
    camera_json = output_root / "camera.json"
    bvh_dir = output_root / "bvh"
    sprites_dir = output_root / "sprites_render_final"
    python = sys.executable
    config_arg = str(config_path)

    start_index = STAGES.index(args.from_stage)
    end_index = STAGES.index(args.to_stage)

    manifest = None
    if split_dir.joinpath("manifest.json").exists():
        manifest = load_manifest(split_dir)
    candidate_upscalers = effective_candidate_upscalers(config, manifest)
    use_dp_selection = len(candidate_upscalers) > 1
    if manifest is not None and len(candidate_upscalers) < len(config.candidate_upscalers):
        total_frames = sum(len(char_entry["frames"]) for char_entry in manifest["characters"])
        print(
            f"Large sheet ({total_frames} frames > {config.large_sheet_frame_threshold}): "
            f"using single upscaler {candidate_upscalers[0]}"
        )

    if start_index <= STAGES.index("split") <= end_index:
        command = [python, "scripts/split_sprite_sheet.py", "--config", config_arg]
        if args.dry_run_split:
            command.append("--dry-run")
        run_command(command)

    if start_index <= STAGES.index("inference") <= end_index:
        if use_dp_selection:
            for candidate_name in candidate_upscalers:
                candidate_output = output_root / f"inference_{candidate_name}"
                run_command(
                    [
                        python,
                        "scripts/process_sprite_frames.py",
                        "--config",
                        config_arg,
                        "--split-dir",
                        str(split_dir),
                        "--output-dir",
                        str(candidate_output),
                        "--device",
                        config.device,
                        "--candidate",
                        candidate_name,
                    ]
                )
        else:
            run_command(
                [
                    python,
                    "scripts/process_sprite_frames.py",
                    "--config",
                    config_arg,
                    "--split-dir",
                    str(split_dir),
                    "--output-dir",
                    str(inference_dir),
                    "--device",
                    config.device,
                ]
            )

    if start_index <= STAGES.index("select") <= end_index:
        if use_dp_selection:
            candidate_specs = [
                f"{name}={output_root / f'inference_{name}'}"
                for name in candidate_upscalers
            ]
            run_command(
                [
                    python,
                    "scripts/select_best_pose.py",
                    "--split-dir",
                    str(split_dir),
                    "--output-dir",
                    str(inference_dp),
                    "--candidate",
                    *candidate_specs,
                ]
            )
        elif inference_dir.exists():
            sync_inference_dir(inference_dir, inference_dp)

    post_select_input = inference_dp if inference_dp.exists() else inference_dir

    if start_index <= STAGES.index("multiview") <= end_index:
        run_command(
            [
                python,
                "scripts/fit_multiview_pose.py",
                "--split-dir",
                str(split_dir),
                "--input-dir",
                str(post_select_input),
                "--selection-report",
                str(inference_dp / "selection_report.json"),
                "--output-dir",
                str(inference_multiview),
                "--device",
                config.device,
            ]
        )

    multiview_input = inference_multiview if inference_multiview.exists() else post_select_input

    if start_index <= STAGES.index("repose") <= end_index:
        run_command(
            [
                python,
                "scripts/repose_with_priors.py",
                "--split-dir",
                str(split_dir),
                "--input-dir",
                str(multiview_input),
                "--output-dir",
                str(inference_reposed),
                "--device",
                config.device,
            ]
        )

    if start_index <= STAGES.index("smooth") <= end_index:
        run_command(
            [
                python,
                "scripts/temporal_smooth_poses.py",
                "--split-dir",
                str(split_dir),
                "--input-dir",
                str(inference_reposed),
                "--output-dir",
                str(inference_smoothed),
                "--device",
                config.device,
            ]
        )

    if start_index <= STAGES.index("camera") <= end_index:
        run_command(
            [
                python,
                "scripts/fit_shared_camera.py",
                "--split-dir",
                str(split_dir),
                "--inference-dir",
                str(inference_smoothed),
                "--faces-source-dir",
                str(inference_dir if inference_dir.exists() else post_select_input),
                "--reposed-dir",
                str(inference_reposed),
                "--output",
                str(camera_json),
                "--device",
                config.device,
            ]
        )

    if start_index <= STAGES.index("render") <= end_index:
        run_command(
            [
                python,
                "scripts/render_final_poses.py",
                "--split-dir",
                str(split_dir),
                "--input-dir",
                str(inference_smoothed),
                "--faces-source-dir",
                str(inference_dir if inference_dir.exists() else post_select_input),
                "--camera",
                str(camera_json),
                "--output-dir",
                str(inference_final),
                "--render-scale",
                str(config.render_scale),
            ]
        )

    if start_index <= STAGES.index("export") <= end_index:
        run_command(
            [
                python,
                "scripts/export_bvh.py",
                "--split-dir",
                str(split_dir),
                "--inference-dir",
                str(inference_final),
                "--output-dir",
                str(bvh_dir),
                "--group-by-sequence",
            ]
        )
        run_command(
            [
                python,
                "scripts/reassemble_mesh_render.py",
                "--split-dir",
                str(split_dir),
                "--inference-dir",
                str(inference_final),
                "--sprites-dir",
                str(sprites_dir),
                "--full-output",
                str(output_root / "reassembled" / "mesh_render_full.png"),
                "--render-scale",
                str(config.render_scale),
            ]
        )
        run_command(
            [
                python,
                "scripts/pose_metrics.py",
                "--split-dir",
                str(split_dir),
                "--inference-dir",
                str(inference_final),
                "--report",
                str(output_root / "pose_metrics_report.json"),
            ]
        )
        run_command(
            [
                python,
                "scripts/compare_mesh_render_reassembly.py",
                "--original",
                str(config.input),
                "--split-dir",
                str(split_dir),
                "--inference-dir",
                str(inference_final),
                "--render-scale",
                str(config.render_scale),
                "--overlay-output",
                str(output_root / "reassembled" / "comparison_overlay.png"),
                "--report-output",
                str(output_root / "reassembled" / "comparison_report.json"),
                "--save-original-upscaled",
                str(output_root / "reassembled" / "original_upscaled.png"),
            ]
        )


if __name__ == "__main__":
    main()
