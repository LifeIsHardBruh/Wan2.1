#!/usr/bin/env bash
set -euo pipefail

#   ./run_cache_1_3B.sh            # run both stages sequentially
#   ./run_cache_1_3B.sh a          # only stage A (record cache, mask optional)
#   ./run_cache_1_3B.sh b          # only stage B (requires mask + existing cache dir)

STAGE="${1:-both}"  # accepted: a, b, both

CKPT_DIR="/data01/henry/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a"
OUT_DIR="/data02/henry/wan_cache"
CACHE_DIR="${OUT_DIR}/cache_data"
PROMPT_A="A brown dog running through a field of flowers"
PROMPT_B="A white cat running through a field of flowers"
# MASK_PATH=""
# MASK_PATH="/data02/henry/wan_cache/masks/mask_step045.pt"  # leave empty if not needed
MASK_PATH="/data02/henry/wan_cache/mask_manual.pt"
OUTPUT_FILENAME="stage_b_manual_mask_reuse_step_10_0.2_blend_noise.mp4"
SAVE_LATENTS=false
LATENT_CACHE_DIR="${OUT_DIR}/latents"
LATENT_SAVE_INTERVAL=5
MASK_SMOOTH_KERNEL="${MASK_SMOOTH_KERNEL:-3}"
MASK_SMOOTH_MODE="${MASK_SMOOTH_MODE:-blend_latent}" #blend_latent/gaussian
MASK_BLEND_LATENT_PATH="/data02/henry/wan_cache/baseline/latents"
SAVE_NOISE="${SAVE_NOISE:-false}"
NOISE_CACHE_DIR="${OUT_DIR}/noises"
NOISE_SAVE_INTERVAL="${NOISE_SAVE_INTERVAL:-1}"
BLUR_MASK_IMAGE_PATH="/data02/henry/wan_cache/mask_blur_image.png"

COMMON_ARGS=(
  --ckpt_dir "${CKPT_DIR}"
  --task t2v-1.3B
  --size 480*832
  --frame_num 81
  --steps 50
  --cfg_scale 5.0
  --prompt_a "${PROMPT_A}"
  --prompt_b "${PROMPT_B}"
  --cache_content full
  --cache_backend disk
  --cache_dir "${CACHE_DIR}"
  --cache_reuse_start_step 10
  --seed 1234
  --out_dir "${OUT_DIR}"
  --stage "${STAGE}"
  --output_filename "${OUTPUT_FILENAME}"
  --device 3
  --mask_smooth_kernel "${MASK_SMOOTH_KERNEL}"
  --mask_smooth_mode "${MASK_SMOOTH_MODE}"
  --mask_blend_latent_path "${MASK_BLEND_LATENT_PATH}"
)

if [[ -n "${MASK_PATH}" && -e "${MASK_PATH}" ]]; then
  COMMON_ARGS+=(--mask "${MASK_PATH}")
fi

if [[ "${STAGE}" == "b" ]]; then
  COMMON_ARGS+=(--activation_cache_path "${CACHE_DIR}")
fi

if [[ "${SAVE_LATENTS}" == "true" ]]; then
  COMMON_ARGS+=(--save_latents --latent_cache_dir "${LATENT_CACHE_DIR}" --latent_save_interval "${LATENT_SAVE_INTERVAL}")
fi

if [[ "${SAVE_NOISE}" == "true" ]]; then
  COMMON_ARGS+=(--save_noise --noise_cache_dir "${NOISE_CACHE_DIR}" --noise_save_interval "${NOISE_SAVE_INTERVAL}")
fi

if [[ -n "${BLUR_MASK_IMAGE_PATH}" ]]; then
  COMMON_ARGS+=(--blur_mask_image_path "${BLUR_MASK_IMAGE_PATH}")
fi

# run from repo root so `wan` package can be imported
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
python -W ignore::FutureWarning -m script.cache_demo "${COMMON_ARGS[@]}"
