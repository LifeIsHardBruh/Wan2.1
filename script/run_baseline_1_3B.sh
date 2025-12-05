#!/usr/bin/env bash
set -euo pipefail

# Baseline run via cache_demo (no caching, single prompt).

CKPT_DIR="/data01/henry/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a"
OUT_DIR="/data02/henry/wan_cache/baseline"
# PROMPT_BASELINE="A white cat running through a field of flowers"
PROMPT_BASELINE="A brown dog running through a field of flowers"
OUTPUT_FILENAME="baseline.mp4"
SAVE_LATENTS=true
LATENT_CACHE_DIR="${OUT_DIR}/latents"
LATENT_SAVE_INTERVAL=1
SAVE_NOISE="${SAVE_NOISE:-true}"
NOISE_CACHE_DIR="${OUT_DIR}/noises"
NOISE_SAVE_INTERVAL="${NOISE_SAVE_INTERVAL:-1}"

mkdir -p "${OUT_DIR}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ARGS=(
  --ckpt_dir "${CKPT_DIR}"
  --task t2v-1.3B
  --size 480*832
  --frame_num 81
  --steps 50
  --cfg_scale 5.0
  --stage baseline
  --prompt_baseline "${PROMPT_BASELINE}"
  --output_filename "${OUTPUT_FILENAME}"
  --seed 1234
  --out_dir "${OUT_DIR}"
)

if [[ "${SAVE_LATENTS}" == "true" ]]; then
  ARGS+=(--save_latents --latent_cache_dir "${LATENT_CACHE_DIR}" --latent_save_interval "${LATENT_SAVE_INTERVAL}")
fi

if [[ "${SAVE_NOISE}" == "true" ]]; then
  ARGS+=(--save_noise --noise_cache_dir "${NOISE_CACHE_DIR}" --noise_save_interval "${NOISE_SAVE_INTERVAL}")
fi

python -W ignore::FutureWarning -m script.cache_demo "${ARGS[@]}"
