#!/usr/bin/env bash
# Optional native OffloadingConnector path (after apply-sm120-overlay.sh).
# Prefer run-offload.sh (SimpleCPU) first; use this only if you need native.
set -euo pipefail

GLM_HOME="${GLM_HOME:-/mnt/data/workspace/glm53}"
VENV="${GLM_HOME}/.venv"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in \
      /usr/local/cuda-13.1 \
      /usr/local/cuda-13 \
      /usr/local/cuda \
      /usr/local/cuda-13.2 \
      /usr/local/cuda-12.8; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      CUDA_HOME="${candidate}"
      break
    fi
  done
fi
export CUDA_HOME="${CUDA_HOME:?error: set CUDA_HOME to a toolkit that contains bin/nvcc}"
export PATH="${CUDA_HOME}/bin:${VENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
unset VLLM_USE_SIMPLE_KV_OFFLOAD || true

FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${GLM_HOME}/vllm_cache/flashinfer_cache}"
export FLASHINFER_WORKSPACE_BASE
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${FLASHINFER_WORKSPACE_BASE}/cubins}"
mkdir -p "${FLASHINFER_CUBIN_DIR}"

MODEL="${MODEL:-RedHatAI/GLM-5.3-Flash-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8102}"
TP="${TP:-4}"
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-400}"
HF_OVERRIDES="${HF_OVERRIDES:-{\"text_config\":{\"index_topk\":2044}}}"

exec "${VENV}/bin/vllm" serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name glm-5.3 \
  --trust-remote-code \
  --tensor-parallel-size "${TP}" \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --async-scheduling \
  --no-enable-flashinfer-autotune \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --kv-cache-dtype fp8 \
  --hf-overrides "${HF_OVERRIDES}" \
  --kv-offloading-size "${KV_OFFLOAD_GIB}" \
  --kv-offloading-backend native
