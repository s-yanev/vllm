#!/usr/bin/env bash
# GLM-5.3-Flash on H200 with recipe SimpleCPUOffloadConnector (kv_offload=simple).
# No --kv-cache-dtype fp8 (Hopper). No OffloadingConnector unless overlaying #54743.
set -euo pipefail

GLM_HOME="${GLM_HOME:-/mnt/data/workspace/glm53}"
VENV="${GLM_HOME}/.venv"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="${CUDA_HOME}/bin:${VENV}/bin:${PATH}"

# ~200 GiB per rank — scale to free host RAM (recipe uses ~220 GiB/rank on large hosts)
CPU_GIB_PER_RANK="${CPU_GIB_PER_RANK:-200}"
CPU_BYTES_PER_RANK=$((CPU_GIB_PER_RANK * 1024**3))

MODEL="${MODEL:-RedHatAI/GLM-5.3-Flash-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8101}"
TP="${TP:-2}"

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
  --kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":${CPU_BYTES_PER_RANK},\"lazy_offload\":false}}"
