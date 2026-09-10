#!/usr/bin/env bash
# GLM-5.3-Flash on 4x RTX PRO 6000 with SimpleCPUOffloadConnector.
# Requires: install-* + apply-sm120-overlay.sh (includes #54756/#54743).
# Same SM120 flags as run.sh, plus CPU KV offload (recipe kv_offload=simple).
# Do NOT use glm53-flash/run.sh on this host.
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

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found (CUDA_HOME=${CUDA_HOME})." >&2
  exit 1
fi
echo "Using CUDA_HOME=${CUDA_HOME} nvcc=$(command -v nvcc)"

FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${GLM_HOME}/vllm_cache/flashinfer_cache}"
export FLASHINFER_WORKSPACE_BASE
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${FLASHINFER_WORKSPACE_BASE}/cubins}"
mkdir -p "${FLASHINFER_CUBIN_DIR}"

# Scale to free host RAM (GiB per TP rank). TP=4 → total ≈ 4 * CPU_GIB_PER_RANK.
CPU_GIB_PER_RANK="${CPU_GIB_PER_RANK:-100}"
CPU_BYTES_PER_RANK=$((CPU_GIB_PER_RANK * 1024**3))

MODEL="${MODEL:-RedHatAI/GLM-5.3-Flash-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8101}"
TP="${TP:-4}"
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
  --kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":${CPU_BYTES_PER_RANK},\"lazy_offload\":false}}"
