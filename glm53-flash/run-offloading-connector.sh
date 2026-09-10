#!/usr/bin/env bash
# Optional OffloadingConnector path (after apply-offloading-overlay.sh).
# Primary serve path remains run.sh (SimpleCPUOffloadConnector).
set -euo pipefail

GLM_HOME="${GLM_HOME:-/mnt/data/workspace/glm53}"
VENV="${GLM_HOME}/.venv"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="${CUDA_HOME}/bin:${VENV}/bin:${PATH}"
# Ensure we do NOT force SimpleCPU when testing native OffloadingConnector
unset VLLM_USE_SIMPLE_KV_OFFLOAD || true

MODEL="${MODEL:-RedHatAI/GLM-5.3-Flash-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8102}"
TP="${TP:-2}"
# Total GiB across ranks for native offload pool
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-400}"

exec "${VENV}/bin/vllm" serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name glm-5.3 \
  --trust-remote-code \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization 0.93 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --enable-expert-parallel \
  --enable-prefix-caching \
  --async-scheduling \
  --no-enable-flashinfer-autotune \
  --kv-offloading-size "${KV_OFFLOAD_GIB}" \
  --kv-offloading-backend native
