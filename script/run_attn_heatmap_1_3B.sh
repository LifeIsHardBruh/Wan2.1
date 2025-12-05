#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   ./run_attn_heatmap_1_3B.sh                                          # run stage A + B, record attn @ step10
#   ATTN_STEPS=9 CACHE_REUSE_START_STEP=50 ./run_attn_heatmap_1_3B.sh      # override env vars for mode/steps

CKPT_DIR="/data01/henry/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a"
OUT_DIR="/data02/henry/wan_cache"
CACHE_DIR="${OUT_DIR}/cache_data"
HEATMAP_DIR="${OUT_DIR}/attn_heatmaps"
PROMPT_A="A brown dog running through a field of flowers"
PROMPT_B="A white cat running through a field of flowers"
MASK_PATH="/data02/henry/wan_cache/mask_manual.pt"
MASK_SMOOTH_KERNEL="${MASK_SMOOTH_KERNEL:-1}"
MASK_SMOOTH_MODE="${MASK_SMOOTH_MODE:-gaussian}"
MASK_BLEND_LATENT_PATH="${MASK_BLEND_LATENT_PATH:-}"
BLUR_MASK_IMAGE_PATH="${BLUR_MASK_IMAGE_PATH:-}"

# Cache reuse start (can override via env)
CACHE_REUSE_START_STEP="${CACHE_REUSE_START_STEP:-2}"

# Attention recording defaults (override via env)
DEFAULT_ATTN_STEP=$((CACHE_REUSE_START_STEP - 1))
if (( DEFAULT_ATTN_STEP < 0 )); then
  DEFAULT_ATTN_STEP=0
fi
ATTN_STEPS="${ATTN_STEPS:-${DEFAULT_ATTN_STEP}}"
ATTN_LAYERS="${ATTN_LAYERS:-last4}"
ATTN_MODE="${ATTN_MODE:-aggregated}"
ATTN_CHUNK="${ATTN_CHUNK:-1024}"

# Optional: reuse existing Stage-A cache when SKIP_STAGE_A=1
SKIP_STAGE_A="${SKIP_STAGE_A:-1}"

COMMON_ARGS=(
  --ckpt_dir "${CKPT_DIR}"
  --task t2v-1.3B
  --size 480*832
  --frame_num 81
  --steps 50
  --cfg_scale 5.0
  --prompt_a "${PROMPT_A}"
  --prompt_b "${PROMPT_B}"
  --mask "${MASK_PATH}"
  --cache_content full
  --cache_backend disk
  --cache_dir "${CACHE_DIR}"
  --cache_reuse_start_step 2
  --seed 1234
  --out_dir "${OUT_DIR}"
  --heatmap_dir "${HEATMAP_DIR}"
  --attn_steps "${ATTN_STEPS}"
  --attn_layers "${ATTN_LAYERS}"
  --attn_mode "${ATTN_MODE}"
  --attn_chunk_size "${ATTN_CHUNK}"
  --output_filename "attn_stage_b.mp4"
  --mask_smooth_kernel "${MASK_SMOOTH_KERNEL}"
  --mask_smooth_mode "${MASK_SMOOTH_MODE}"
  --mask_blend_latent_path "${MASK_BLEND_LATENT_PATH}"
  --device 3
)

if [[ "${SKIP_STAGE_A}" == "1" ]]; then
  COMMON_ARGS+=(--skip_stage_a --activation_cache_path "${CACHE_DIR}")
fi

if [[ -n "${BLUR_MASK_IMAGE_PATH}" ]]; then
  COMMON_ARGS+=(--blur_mask_image_path "${BLUR_MASK_IMAGE_PATH}")
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
python -W ignore::FutureWarning -m script.attn_heatmap "${COMMON_ARGS[@]}"
