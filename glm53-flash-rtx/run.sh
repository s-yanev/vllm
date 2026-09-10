#!/usr/bin/env bash
# GLM-5.3-Flash on 4x RTX PRO 6000 (SM120).
# Requires: install-nightly-wheel.sh + install-flashinfer.sh + apply-sm120-overlay.sh
# FP8 KV (Blackwell). No CPU offload in the primary path.
# Do NOT use glm53-flash/run.sh (H200 TP=2 / BF16 / SimpleCPU) on this host.
#
# DeepGEMM MHC JIT (mhc_pre_tilelang / tf32_hc_prenorm_gemm) requires a real
# nvcc on PATH — flashinfer-cubin alone is not enough for that path.
set -euo pipefail

GLM_HOME="${GLM_HOME:-/mnt/data/workspace/glm53}"
VENV="${GLM_HOME}/.venv"

# Prefer an existing CUDA toolkit with nvcc. Hosts often lack cuda-13.2 even when
# cuda-13.1 / /usr/local/cuda are installed (DeepGEMM JIT needs nvcc on PATH).
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

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found (CUDA_HOME=${CUDA_HOME})." >&2
  echo "  DeepGEMM MHC JIT needs the CUDA toolkit compiler." >&2
  echo "  Fix: export CUDA_HOME=/path/to/cuda && export PATH=\$CUDA_HOME/bin:\$PATH" >&2
  echo "  Probe: ls /usr/local/cuda*/bin/nvcc" >&2
  exit 1
fi
echo "Using CUDA_HOME=${CUDA_HOME} nvcc=$(command -v nvcc) ($(nvcc --version | awk '/release/ {print $6}' | tr -d ','))"

# FlashInfer caches (override if your host layout differs)
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${GLM_HOME}/vllm_cache/flashinfer_cache}"
export FLASHINFER_WORKSPACE_BASE
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${FLASHINFER_WORKSPACE_BASE}/cubins}"
mkdir -p "${FLASHINFER_CUBIN_DIR}"

MODEL="${MODEL:-RedHatAI/GLM-5.3-Flash-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8101}"
TP="${TP:-4}"
# RedHatAI ships index_topk=2048 → buffer 2176; FlashInfer SM120 GLM_NSA
# kernels only instantiate topk=2048. Official layout uses 2044 → 2048.
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
  --hf-overrides "${HF_OVERRIDES}"
