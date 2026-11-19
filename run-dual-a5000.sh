#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MAIN_GPU="${MAIN_GPU:-0}"
export SUMMARY_GPU="${SUMMARY_GPU:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAIN_GPU_MEMORY_UTILIZATION="${MAIN_GPU_MEMORY_UTILIZATION:-0.95}"
export MAIN_ENFORCE_EAGER="${MAIN_ENFORCE_EAGER:-1}"
export MAIN_MAX_NUM_SEQS="${MAIN_MAX_NUM_SEQS:-1}"
export MAIN_KV_CACHE_DTYPE="${MAIN_KV_CACHE_DTYPE:-fp8}"

CHUNK=1 \
ENABLE_SUMMARIZER=0 \
KEEP_QA_HISTORY=0 \
LIVE_VLM_PROCESS_INTERVAL=3 \
LIVE_VLM_FRAMES_PER_BATCH=12 \
exec bash "${SCRIPT_DIR}/services/scripts/run.sh" minimal --no-ssl "$@"
