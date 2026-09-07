#!/usr/bin/env bash
# Generate BVH animations for every townsfolk character row in input/townsfolk.png.
#
# Usage:
#   ./townfolks.sh
#   INPUT=input/townsfolk.png OUTPUT_ROOT=output/townsfolk ./townfolks.sh
#   ./townfolks.sh --no-upscale
#
# Each character is written to:
#   $OUTPUT_ROOT/char_XX/animation.bvh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

INPUT="${INPUT:-input/townsfolk.png}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/townsfolk}"
CONFIG="${CONFIG:-configs/townsfolk.json}"
EXTRA_ARGS=()
USE_UPSCALE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-upscale)
            USE_UPSCALE=0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

if [[ "$USE_UPSCALE" == "1" ]]; then
    EXTRA_ARGS+=(--upscale)
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

if [[ ! -f "$INPUT" ]]; then
    echo "error: input sprite sheet not found: $INPUT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "Townsfolk BVH batch"
echo "  input:  $INPUT"
echo "  output: $OUTPUT_ROOT"
echo

mapfile -t CHAR_ROWS < <(
    python3 scripts/townsfolk_rows.py --input "$INPUT" --config "$CONFIG"
)

if [[ ${#CHAR_ROWS[@]} -eq 0 ]]; then
    echo "error: no townsfolk characters detected" >&2
    exit 1
fi

for row in "${CHAR_ROWS[@]}"; do
    IFS='|' read -r name x y width height frame_count is_loop <<<"$row"
    rect="${x},${y},${width},${height}"
    output_dir="${OUTPUT_ROOT}/${name}"
    loop_flag=()
    if [[ "$is_loop" == "1" ]]; then
        loop_flag=(--loop)
    fi

    echo "=== ${name}: ${frame_count} frames, rect=${rect}, loop=${is_loop} ==="
    python scripts/basic_pipeline.py \
        --image "$INPUT" \
        --rect "$rect" \
        --frames "$frame_count" \
        --output "$output_dir" \
        "${loop_flag[@]}" \
        "${EXTRA_ARGS[@]}"
    echo
done

echo "Done. BVH files:"
for row in "${CHAR_ROWS[@]}"; do
    IFS='|' read -r name _ _ _ _ _ _ <<<"$row"
    echo "  ${OUTPUT_ROOT}/${name}/animation.bvh"
done
